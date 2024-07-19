import cspace.cspace.classes
import cspace.torch.classes
import transformers
import accelerate
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
    def __init__(self, data, pose, state):
        assert isinstance(pose, cspace.torch.classes.LinkPoseCollection)
        assert len(pose.batch) == 1
        assert isinstance(state, cspace.torch.classes.JointStateCollection)
        assert len(state.batch) == 1
        assert pose.batch == state.batch
        assert data.shape[:-2] == pose.batch

        self._data_ = data
        self._pose_ = pose
        self._state_ = state

    def __len__(self):
        return self._pose_.batch[0]

    def __getitem__(self, key):
        data = self._data_[key : key + 1, ...]
        pose = cspace.torch.classes.LinkPoseCollection(
            base=self._pose_.base,
            name=self._pose_.name,
            position=torch.select(self._pose_._position_, dim=0, index=key),
            orientation=torch.select(self._pose_._orientation_, dim=0, index=key),
        )
        state = cspace.torch.classes.JointStateCollection(
            name=self._state_.name,
            position=torch.select(self._state_._position_, dim=0, index=key),
        )
        return data, pose, state

    @classmethod
    def collate_fn(cls, entries):
        data, pose, state = list(zip(*entries))

        info = {(e.base, e.name) for e in pose}
        assert len(info) == 1
        base, name = next(iter(info))
        position = torch.stack(tuple(e._position_ for e in pose), dim=0)
        orientation = torch.stack(tuple(e._orientation_ for e in pose), dim=0)

        pose = cspace.torch.classes.LinkPoseCollection(
            base=base,
            name=name,
            position=position,
            orientation=orientation,
        )

        info = {e.name for e in state}
        assert len(info) == 1
        name = next(iter(info))
        position = torch.stack(tuple(e._position_ for e in state), dim=0)

        state = cspace.torch.classes.JointStateCollection(
            name=name,
            position=position,
        )

        data = torch.concatenate(data, dim=0)

        return data, pose, state


class InverseKinematics(cspace.torch.classes.Kinematics):
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self, description, *link, base=None, model=None, bucket=None):
        super().__init__(description, *link, base=base)
        if model:
            self.bucket = bucket if bucket else 1000
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (6 * len(self.link)) * self.bucket,
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

    def encode(self, pose):
        zero = self.forward(
            cspace.torch.classes.JointStateCollection.zero(
                self.spec, self.joint, pose.batch
            )
        )
        delta = zero.delta(self.spec, pose)

        delta = torch.reshape(delta, pose.batch + tuple([-1]))

        value = delta * (self.bucket - 1)
        value = torch.clip(value.to(torch.int64), min=0, max=self.bucket - 1)
        encoded = torch.nn.functional.one_hot(value, self.bucket)
        encoded = torch.flatten(encoded, -2, -1)
        encoded = torch.unsqueeze(encoded, -2)
        return encoded

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

    def optimize(self, *, dataloader, optimizer, scheduler, epoch, save):
        epoch_total = epoch

        accelerator = accelerate.Accelerator()
        accelerate.logging.get_logger(__name__).info(
            "[Train] ----- Dataset: {} (epoch={}, batch={}) - creation".format(
                len(dataloader.dataset), epoch_total, dataloader.batch_size
            )
        )

        dataloader, model, optimizer, scheduler = accelerator.prepare(
            dataloader, self.model, optimizer, scheduler
        )

        model.train()
        for epoch in range(epoch_total):
            total, count = 0, 0
            for batch, (data, pose, true) in enumerate(dataloader):
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
                accelerate.logging.get_logger(__name__).info(
                    "[Train] ----- Epoch {} [{}/{}] - Loss: {} [/Train]".format(
                        epoch,
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
        accelerate.logging.get_logger(__name__).info(
            "[Train] ----- Dataset: {} (epoch={}, batch={}) - complete".format(
                len(dataloader.dataset), epoch_total, dataloader.batch_size
            )
        )

    def loss(self, pred, true):
        assert true.batch == pred.shape[:-1]

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

    def train(
        self, *, total=None, epoch=None, batch=None, noise=None, seed=None, save=None
    ):
        epoch = epoch if epoch else 1
        batch_size = batch if batch else 128
        entry_total = total if total else 1024

        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )

        dataset = InverseDataset(
            *self.rand(
                total=entry_total,
                noise=noise,
                seed=seed,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=InverseDataset.collate_fn,
            shuffle=True,
        )

        self.optimize(
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            save=save,
        )

    def rand(self, *, total, noise, seed=None, std=0.01):
        generator = torch.Generator().manual_seed(seed)
        zero = cspace.torch.classes.JointStateCollection.zero(
            self.spec, self.joint, batch=[total]
        )
        state = zero.apply(
            self.spec,
            torch.rand(
                total, len(self.joint), generator=generator, dtype=torch.float64
            ),
        )
        pose = self.forward(state)

        if noise:
            linear_shape = [noise] + list(pose.batch) + [3, len(pose.name)]

            linear_std = torch.ones(linear_shape, dtype=torch.float64) * std
            linear_mean = torch.zeros(linear_shape, dtype=torch.float64)

            linear_noise = torch.normal(linear_mean, linear_std, generator=generator)

            position = pose._position_
            linear = position.expand(*linear_shape)
            linear = linear + linear_noise
            position = linear

            angular_shape = [noise] + list(pose.batch) + [3, len(pose.name)]

            angular_std = torch.ones(angular_shape, dtype=torch.float64) * std
            angular_mean = torch.zeros(angular_shape, dtype=torch.float64)

            angular_noise = torch.normal(angular_mean, angular_std, generator=generator)

            orientation = pose._orientation_
            angular = torch.transpose(
                cspace.torch.ops.qua_to_rpy(torch.transpose(orientation, -2, -1)),
                -2,
                -1,
            )
            angular = angular + angular_noise
            orientation = torch.transpose(
                cspace.torch.ops.rpy_to_qua(torch.transpose(angular, -2, -1)), -2, -1
            )

            pose = cspace.torch.classes.LinkPoseCollection(
                base=pose.base,
                name=pose.name,
                position=torch.flatten(position, 0, 1),
                orientation=torch.flatten(orientation, 0, 1),
            )

            shape = [noise] + list(state.batch) + [len(state.name)]

            position = state._position_
            position = position.expand(*shape)

            state = cspace.torch.classes.JointStateCollection(
                name=state.name,
                position=torch.flatten(position, 0, 1),
            )
        data = self.encode(pose)
        return data, pose, state


class PolicyKinematics(cspace.torch.classes.Kinematics):
    def __init__(self, description, *link, base=None, model=None):
        super().__init__(description, *link, base=base)
        if model:
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (6 * len(self.link)) * self.bucket,
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
