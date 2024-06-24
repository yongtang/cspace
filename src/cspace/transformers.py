import cspace.cspace.classes
import cspace.torch.classes
import transformers
import accelerate
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pose, state):
        assert isinstance(pose, cspace.torch.classes.LinkPoseCollection)
        assert len(pose.batch) == 1
        assert isinstance(state, cspace.torch.classes.JointStateCollection)
        assert len(state.batch) == 1
        assert pose.batch == state.batch

        self._pose_ = pose
        self._state_ = state

    def __len__(self):
        return self._pose_.batch[0]

    def __getitem__(self, key):
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
        return pose, state

    @classmethod
    def collate_fn(cls, entries):
        pose, state = list(zip(*entries))

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

        return pose, state


class Model(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.embedding = torch.nn.LazyLinear(
            transformer.get_input_embeddings().embedding_dim,
            #   device=transformer.get_input_embeddings().weight.device,
            dtype=transformer.get_input_embeddings().weight.dtype,
            bias=False,
        )
        transformer.get_input_embeddings().reset_parameters()
        for param in transformer.get_input_embeddings().parameters():
            param.requires_grad = False
        transformer.get_output_embeddings().reset_parameters()
        for param in transformer.get_output_embeddings().parameters():
            param.requires_grad = False
        self.transformer = transformer

    def forward(self, batch):
        data = list(
            map(
                lambda e: e.to(self.embedding.weight.dtype).to(
                    self.embedding.weight.device
                ),
                batch,
            )
        )
        mask = list(map(lambda e: torch.ones(len(e), device=e.device), data))

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        data = self.transformer(
            inputs_embeds=self.embedding(data),
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        data = torch.stack(
            [data[i, j] for i, j in enumerate(torch.count_nonzero(mask, dim=1) - 1)]
        )

        data = torch.mm(data, self.embedding.weight)
        return data


class Kinematics(cspace.torch.classes.Kinematics):
    bucket: int = 1000
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self, description, *link, base=None, model=None):
        super().__init__(description, *link, base=base, model=model)
        if model:
            self.model = Model(
                transformer=transformers.AutoModelForCausalLM.from_pretrained(model)
            )
            # initialize parameters
            self.inverse(
                self.forward(
                    cspace.torch.classes.JointStateCollection.zero(
                        self.spec, self.joint
                    )
                )
            )

    def inverse(self, pose):

        with torch.no_grad():
            data = self.head(pose)

            batch = list(data.shape[:-2])

            data = torch.reshape(data, [-1] + list(data.shape[-2:]))

            pred = self.model(data)

            pred = torch.reshape(pred, batch + list(pred.shape[-1:]))

            encoded = torch.unflatten(pred, -1, (-1, self.bucket))
            assert encoded.shape[-2:] == torch.Size(
                [6 * len(self.link) + 1 * len(self.joint), self.bucket]
            )
            encoded = encoded[..., 6 * len(self.link) :, :]

            delta_value = torch.argmax(encoded, dim=-1)
            delta_value = delta_value.to(torch.float64) / (self.bucket - 1)
            delta_value = torch.clip(delta_value, min=0.0, max=1.0)

            zero = cspace.torch.classes.JointStateCollection.zero(
                self.spec, self.joint, batch
            )
            return zero.apply(self.spec, delta_value)

    def loss(self, pred, true):
        assert true.batch == pred.shape[:-1]

        pred_value = torch.unflatten(pred, -1, (-1, self.bucket))
        assert pred_value.shape[-2:] == torch.Size(
            [6 * len(self.link) + 1 * len(self.joint), self.bucket]
        )
        pred_value = pred_value[..., 6 * len(self.link) :, :]
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

    def head(self, pose):
        zero = self.forward(
            cspace.torch.classes.JointStateCollection.zero(
                self.spec, self.joint, pose.batch
            )
        )
        delta = zero.delta(self.spec, pose)
        blank = torch.zeros(pose.batch + (1, len(self.joint)))

        delta = torch.reshape(delta, pose.batch + tuple([-1]))
        blank = torch.reshape(blank, pose.batch + tuple([-1]))

        value = torch.concatenate((delta, blank), dim=-1)
        value = value * (self.bucket - 1)
        value = torch.clip(value.to(torch.int64), min=0, max=self.bucket - 1)
        encoded = torch.nn.functional.one_hot(value, self.bucket)
        encoded = torch.flatten(encoded, -2, -1)
        encoded = torch.unsqueeze(encoded, -2)
        return encoded

    def rand(self, total, noise, seed=None, std=0.01):
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

        return pose, state

    def train(
        self,
        total=None,
        epoch=None,
        batch=None,
        noise=None,
        seed=None,
        save=None,
    ):
        entry_total = total if total else 1024
        epoch_total = epoch if epoch else 1
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

        accelerator = accelerate.Accelerator()

        accelerate.logging.get_logger(__name__).info(
            "[Train] ----- Dataset: {} (entry={}, noise={}, batch={}, seed={}) - creation".format(
                entry_total * (noise if noise else 1),
                entry_total,
                (noise if noise else 1),
                batch_size,
                seed,
            )
        )

        dataset = Dataset(*self.rand(total=entry_total, noise=noise, seed=seed))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=Dataset.collate_fn,
            shuffle=True,
        )

        accelerate.logging.get_logger(__name__).info(
            "[Train] ----- Dataset: {} (entry={}, noise={}, batch={}, seed={}) - complete".format(
                len(dataset), entry_total, (noise if noise else 1), batch_size, seed
            )
        )

        dataloader, model, optimizer, scheduler = accelerator.prepare(
            dataloader, self.model, optimizer, scheduler
        )

        model.train()
        for epoch in range(epoch_total):
            total, count = 0, 0
            for batch, (pose, true) in enumerate(dataloader):
                data = self.head(pose)
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
                        len(dataset),
                        total / count,
                    )
                )
            accelerator.save_state(save) if save else None
