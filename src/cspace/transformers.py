import cspace.cspace.classes
import cspace.torch.classes
import transformers
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
    def __init__(self, joint, link, bucket, length, total, noise=None):
        total = total if noise is None else (noise * total)

        self.index = torch.randint(
            low=0, high=bucket, size=(total, length, len(joint)), dtype=torch.int64
        )
        if not noise:
            self.noise = torch.zeros((total, len(link), 6), dtype=torch.float64)
        else:
            std = torch.tensor(0.01, dtype=torch.float64).expand(total, len(link), 6)
            mean = torch.tensor(0.0, dtype=torch.float64).expand(total, len(link), 6)
            self.noise = torch.normal(mean, std)

        prod = torch.cumprod(
            torch.tensor(1.0 / bucket, dtype=torch.float64).expand([1, length, 1]),
            dim=-2,
        )
        zero = prod * bucket / 2.0

        self.scale = torch.sub(
            torch.add(torch.multiply(self.index, prod), prod / 2), zero
        )

        self.scale = torch.concatenate(
            (
                torch.zeros((total, 1, len(joint)), dtype=torch.float64),
                self.scale,
            ),
            dim=-2,
        )

        self.scale = torch.cumsum(self.scale, dim=-2)  # (-0.5, 0.5)

        self.scale = self.scale * 2.0  # (-1.0, 1.0)

    def __len__(self):
        return self.scale.shape[0]

    def __getitem__(self, key):
        return (self.scale[key], self.index[key], self.noise[key])


class InverseKinematics(cspace.torch.classes.Kinematics):
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(
        self, description, *link, base=None, model=None, bucket=None, length=None
    ):
        super().__init__(description, *link, base=base)
        if model:
            self.bucket = bucket if bucket else 10
            self.length = length if length else 3
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (len(self.joint) * 1 + len(self.link) * 6),
                transformer.get_input_embeddings().embedding_dim,
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )
            output_embeddings = torch.nn.Linear(
                transformer.get_input_embeddings().embedding_dim,
                (len(self.joint) * self.bucket),
                dtype=transformer.get_output_embeddings().weight.dtype,
                bias=False,
            )

            self.model = Model(transformer, input_embeddings, output_embeddings)

    def inverse(self, pose):

        repeat = 16

        with torch.no_grad():
            position = torch.stack(
                tuple(pose.position(name) for name in pose.name), dim=-1
            )
            orientation = torch.stack(
                tuple(pose.orientation(name) for name in pose.name), dim=-1
            )

            position = position.unsqueeze(-3).expand(
                *(pose.batch + (repeat, 3, len(pose.name)))
            )
            orientation = orientation.unsqueeze(-3).expand(
                *(pose.batch + (repeat, 4, len(pose.name)))
            )

            transform = cspace.torch.classes.Transform(
                xyz=torch.transpose(position, -1, -2),
                rot=cspace.torch.ops.qua_to_rot(torch.transpose(orientation, -1, -2)),
            )

            task = cspace.torch.classes.LinkPoseCollection(
                pose.base, pose.name, position, orientation
            )

            state = [
                cspace.torch.classes.JointStateCollection.apply(
                    self.spec,
                    self.joint,
                    torch.zeros(task.batch + tuple([len(self.joint)])),
                    min=-1.0,
                    max=1.0,
                )
            ]

            for step in range(self.length):
                data = self.encode(state, task)

                pred = self.model(data)

                state.append(self.decode(state, pred))

            state = state[-1]

            task = self.forward(state)

            position = torch.stack(
                tuple(task.position(name) for name in pose.name), dim=-1
            )
            orientation = torch.stack(
                tuple(task.orientation(name) for name in pose.name), dim=-1
            )

            measure = (
                transform.inverse()
                * cspace.torch.classes.Transform(
                    xyz=torch.transpose(position, -1, -2),
                    rot=cspace.torch.ops.qua_to_rot(
                        torch.transpose(orientation, -1, -2)
                    ),
                )
            ).log

            loss = torch.sqrt(
                torch.sum(torch.square(torch.flatten(measure, -2, -1)), dim=-1)
            )

            selection = torch.min(loss, dim=-1)
            selection = torch.reshape(selection.indices, [-1])

            position = torch.stack(
                tuple(state.position(self.spec, name) for name in state.name), dim=-1
            )
            position = torch.reshape(position, (-1, repeat, len(self.joint)))
            position = torch.index_select(position, dim=-2, index=selection)
            position = torch.reshape(position, pose.batch + tuple([len(self.joint)]))

            state = cspace.torch.classes.JointStateCollection(self.joint, position)

            return state

    def train(
        self,
        *,
        logger,
        accelerator,
        save=None,
        batch=None,
        total=None,
        repeat=None,
        lr=None,
        noise=None,
    ):
        batch = batch if batch else 128
        total = total if total else 1024
        repeat = repeat if repeat else 1
        lr = lr if lr else 1e-5

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )

        for r in range(repeat):
            logger.info(
                "[Train] ----- Dataset: (r={}/{}, total={}, batch={}) - creation".format(
                    r, repeat, total, batch
                )
            )
            dataset = cspace.transformers.InverseDataset(
                self.joint,
                self.link,
                self.bucket,
                self.length,
                total,
                noise=noise,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch,
                shuffle=True,
            )

            dataloader, model, optimizer, scheduler = accelerator.prepare(
                dataloader, self.model, optimizer, scheduler
            )
            model.train()

            loss_total, loss_count = 0, 0
            for b, (scale, true, delta) in enumerate(dataloader):
                state = tuple(
                    cspace.torch.classes.JointStateCollection.apply(
                        self.spec,
                        self.joint,
                        torch.select(scale, dim=-2, index=step),
                        min=-1.0,
                        max=1.0,
                    )
                    for step in range(self.length)
                )
                task = self.forward(
                    cspace.torch.classes.JointStateCollection.apply(
                        self.spec,
                        self.joint,
                        torch.select(scale, dim=-2, index=self.length),
                        min=-1.0,
                        max=1.0,
                    )
                )
                for step in range(self.length):
                    data = self.encode(state[0 : step + 1], task, delta)
                    pred = model(data)
                    loss = self.loss_fn(
                        torch.unflatten(pred, -1, (self.bucket, -1)),
                        torch.select(true, dim=-2, index=step),
                    )
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    loss = accelerator.gather_for_metrics(loss)
                    pred = accelerator.gather_for_metrics(pred)
                    loss_total += loss.sum().item()
                    loss_count += len(pred)
                logger.info(
                    "[Train] ----- Dataset: (r={}/{}, total={}, batch={}) - b={} - Loss: {}".format(
                        r, repeat, total, batch, b, loss_total / loss_count
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
                "[Train] ----- Dataset: (r={}/{}, total={}, batch={}) - complete".format(
                    r, repeat, total, batch
                )
            )

    def encode(self, state, task, noise=None):
        def f_value(self, entry, task, noise):
            def f_task(mark, task, noise, index, name):
                entry = mark.transform(name).inverse() * task.transform(name)
                if noise is not None:
                    xyz, rot = cspace.torch.ops.se3_exp(
                        torch.select(noise, dim=-2, index=index)
                    )
                    entry = entry * cspace.torch.classes.Transform(xyz=xyz, rot=rot)
                return entry.log

            assert entry.batch == task.batch, "{} vs. {}".format(
                entry.batch, task.batch
            )

            mark = self.forward(entry)

            value = tuple(
                torch.unsqueeze(entry.position(self.spec, name), -1)
                for name in entry.name
            ) + tuple(
                f_task(mark, task, noise, index, name)
                for index, name in enumerate(task.name)
            )

            value = tuple(entry.to(next(iter(value)).device) for entry in value)

            value = torch.concatenate(value, dim=-1)

            value = torch.reshape(value, task.batch + (1, -1))

            return value

        value = tuple(f_value(self, entry, task, noise) for entry in state)

        value = tuple(entry.to(next(iter(value)).device) for entry in value)

        return torch.concatenate(value, dim=-2)

    def decode(self, state, pred):
        pred = torch.unflatten(pred, -1, (self.bucket, -1))
        assert pred.shape[-2:] == (self.bucket, len(self.joint))

        index = torch.argmax(pred, dim=-2)

        prod = torch.cumprod(
            torch.tensor(1.0 / self.bucket, dtype=torch.float64).expand([len(state)]),
            dim=0,
        )[-1]
        zero = prod * self.bucket / 2.0

        scale = (
            state[-1].scale(self.spec, min=-1.0, max=1.0).to(index.device)
            + torch.sub(torch.add(torch.multiply(index, prod), prod / 2), zero) * 2.0
        )  # (-1.0, 1.0)

        state = cspace.torch.classes.JointStateCollection.apply(
            self.spec, self.joint, scale, min=-1.0, max=1.0
        )
        return state


class PolicyKinematics(cspace.torch.classes.Kinematics):
    def __init__(self, description, *link, base=None, model=None):
        super().__init__(description, *link, base=base)
        if model:
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (len(self.joint) * 1 + len(self.link) * 6),
                transformer.get_input_embeddings().embedding_dim,
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )
            output_embeddings = torch.nn.Linear(
                transformer.get_input_embeddings().embedding_dim,
                (len(self.joint) * self.bucket),
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
