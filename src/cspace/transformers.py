import cspace.cspace.classes
import cspace.torch.classes
import transformers
import functools
import torch


class Model(torch.nn.Module):
    def __init__(self, transformer, input_embeddings, output_embeddings):
        super().__init__()
        transformer.get_input_embeddings().reset_parameters()
        for param in transformer.get_input_embeddings().parameters():
            param.requires_grad = False
        transformer.get_output_embeddings().reset_parameters()
        for param in transformer.get_output_embeddings().parameters():
            param.requires_grad = False

        self.transformer = transformer
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings

    def forward(self, data):
        batch = list(data.shape[:-2])

        data = torch.reshape(data, [-1] + list(data.shape[-2:]))

        data = list(
            map(
                lambda e: e.to(self.input_embeddings.weight.dtype).to(
                    self.input_embeddings.weight.device
                ),
                data,
            )
        )
        mask = list(map(lambda e: torch.ones(len(e), device=e.device), data))

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        data = self.transformer(
            inputs_embeds=self.input_embeddings(data),
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        data = torch.stack(
            [data[i, j] for i, j in enumerate(torch.count_nonzero(mask, dim=1) - 1)]
        )

        data = self.output_embeddings(data)

        data = torch.reshape(data, batch + list(data.shape[-1:]))

        return data


class InverseDataset(torch.utils.data.Dataset):
    def __init__(self, total, joint, link, noise=None, seed=None):
        generator = torch.Generator().manual_seed(seed)
        self.delta = torch.rand(
            total, len(joint), generator=generator, dtype=torch.float64
        )
        if not noise:
            self.noise = torch.zeros((total, len(link), 6), dtype=torch.float64)
        else:
            self.delta = self.delta.unsqueeze(0).expand(noise, -1, -1).flatten(0, 1)
            std = torch.tensor(0.01, dtype=torch.float64).expand(
                noise * total, len(link), 6
            )
            mean = torch.tensor(0.0, dtype=torch.float64).expand(
                noise * total, len(link), 6
            )
            self.noise = torch.normal(mean, std, generator=generator)

    def __len__(self):
        return self.delta.shape[0]

    def __getitem__(self, key):
        return (self.delta[key], self.noise[key])


class InverseKinematics(cspace.torch.classes.Kinematics):
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self, description, *link, base=None, model=None, bucket=None):
        super().__init__(description, *link, base=base)
        if model:
            self.bucket = bucket if bucket else 1000
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (6 * len(self.link)),
                transformer.get_input_embeddings().embedding_dim,
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )
            output_embeddings = torch.nn.Linear(
                transformer.get_input_embeddings().embedding_dim,
                (1 * len(self.joint)) * self.bucket,
                dtype=transformer.get_output_embeddings().weight.dtype,
                bias=False,
            )

            self.model = Model(transformer, input_embeddings, output_embeddings)

    def inverse(self, pose):
        with torch.no_grad():
            data = self.encode(pose)

            pred = self.model(data)

            state = self.decode(pred)

            return state

    def train(self, *, logger, accelerator, dataset, batch=None, epoch=None, save=None):
        epoch = epoch if epoch else 1
        batch_size = batch if batch else 128

        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )

        logger.info(
            "[Train] ----- Dataset: {} (epoch={}, batch={}) - creation".format(
                len(dataset), epoch, batch_size
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        dataloader, model, optimizer, scheduler = accelerator.prepare(
            dataloader, self.model, optimizer, scheduler
        )

        zero = cspace.torch.classes.JointStateCollection.zero(
            self.spec, self.joint, batch=[batch_size]
        )

        model.train()
        for index in range(epoch):
            total, count = 0, 0
            for batch, (delta, noise) in enumerate(dataloader):
                true = zero.apply(self.spec, delta)
                pose = self.forward(true)
                data = self.encode(pose, noise)
                pred = model(data)
                loss = self.loss(pred, true)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss = accelerator.gather_for_metrics(loss)
                pred = accelerator.gather_for_metrics(pred)
                total += loss.sum().item()
                count += len(pred)
                logger.info(
                    "[Train] ----- Epoch {} [{}/{}] - Loss: {} [/Train]".format(
                        index,
                        count,
                        len(dataloader.dataset),
                        total / count,
                    )
                )
            (
                accelerator.save(
                    self,
                    save,
                )
                if save
                else None
            )
        logger.info(
            "[Train] ----- Dataset: {} (epoch={}, batch={}) - complete".format(
                len(dataloader.dataset), epoch, dataloader.batch_size
            )
        )

    @functools.cache
    def f_base(self, batch):
        zero = cspace.torch.classes.JointStateCollection.zero(
            self.spec, self.joint, batch=batch
        )
        base = self.forward(zero)
        return base

    def encode(self, pose, noise=None):

        base = self.f_base(pose.batch)

        value = torch.stack(
            tuple(
                (base.transform(name).inverse() * pose.transform(name)).log
                for name in pose.name
            ),
            dim=-2,
        )
        value = value if noise is None else value + noise

        value = torch.flatten(value, 1, -1)
        value = torch.unsqueeze(value, -2)

        return value

    def decode(self, pred):
        batch = pred.shape[:-1]

        encoded = torch.unflatten(pred, -1, (-1, self.bucket))
        assert encoded.shape[-2:] == torch.Size([1 * len(self.joint), self.bucket])

        delta_value = torch.argmax(encoded, dim=-1)
        delta_value = delta_value.to(torch.float64) / (self.bucket - 1)
        delta_value = torch.clip(delta_value, min=0.0, max=1.0)

        zero = cspace.torch.classes.JointStateCollection.zero(
            self.spec, self.joint, batch
        )
        state = zero.apply(self.spec, delta_value)

        return state

    def loss(self, pred, true):
        assert true.batch == pred.shape[:-1], "{} vs. {}".format(true.batch, pred.shape)

        pred_value = torch.unflatten(pred, -1, (-1, self.bucket))
        assert pred_value.shape[-2:] == torch.Size([1 * len(self.joint), self.bucket])
        pred_value = torch.transpose(pred_value, -1, -2)

        zero_state = cspace.torch.classes.JointStateCollection.zero(
            self.spec, self.joint, true.batch
        )
        true_state = cspace.torch.classes.JointStateCollection(
            self.joint,
            torch.stack(
                tuple(true.position(self.spec, name) for name in self.joint), dim=-1
            ),
        )
        true_delta = zero_state.delta(self.spec, true_state)
        true_delta = true_delta.to(pred_value.device)
        true_value = true_delta * (self.bucket - 1)
        true_value = torch.clip(true_value.to(torch.int64), min=0, max=self.bucket - 1)
        return self.loss_fn(pred_value, true_value)


class PolicyKinematics(cspace.torch.classes.Kinematics):
    def __init__(self, description, *link, base=None, model=None):
        super().__init__(description, *link, base=base)
        if model:
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (6 * len(self.link)),
                transformer.get_input_embeddings().embedding_dim,
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )
            output_embeddings = torch.nn.Linear(
                transformer.get_input_embeddings().embedding_dim,
                (1 * len(self.joint)) * self.bucket,
                dtype=transformer.get_output_embeddings().weight.dtype,
                bias=False,
            )

            self.model = Model(transformer, input_embeddings, output_embeddings)

    def policy(self, state, observation):
        with torch.no_grad():
            data = self.encode(state, observation)

            pred = self.model(data)

            state = self.decode(pred)

            return state

    def encode(self, state, observation):
        raise NotImplementedError

    def decode(self, pred):
        raise NotImplementedError
