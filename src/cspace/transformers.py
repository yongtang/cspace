import cspace.cspace.classes
import cspace.torch.classes
import transformers
import PIL.Image
import itertools
import functools
import pathlib
import torch
import json
import abc
import io


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

    def forward(self, data, mask):
        batch = data.shape[:-2]
        assert data.shape[:-1] == mask.shape
        data = torch.reshape(data, [-1] + list(data.shape[-2:]))
        mask = torch.reshape(mask, [-1] + list(mask.shape[-1:]))

        data = data.to(self.input_embeddings.weight.dtype)
        data = data.to(self.input_embeddings.weight.device)
        mask = mask.to(self.input_embeddings.weight.device)

        data = self.transformer(
            inputs_embeds=self.input_embeddings(data),
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        data = torch.stack(
            [data[i, j] for i, j in enumerate(torch.count_nonzero(mask, dim=1) - 1)]
        )

        data = self.output_embeddings(data)

        data = torch.reshape(data, batch + data.shape[-1:])
        return data


class JointStateEncoding:
    def decode(self, state, pred, choice):
        pred = torch.unflatten(pred, -1, (self.bucket, -1))
        assert pred.shape[-2:] == (self.bucket, len(self.joint))

        prod = self.prod[len(state) - 1]

        index = choice(pred)

        scale = state[-1].scale(self.spec, min=0.0, max=1.0)
        scale = scale + torch.multiply(index, prod)
        scale = scale % 1.0

        state = cspace.torch.classes.JointStateCollection.apply(
            self.spec, self.joint, scale, min=0.0, max=1.0
        )
        return state

    def bucketize(self, state):
        value = state.scale(spec=self.spec, min=0.0, max=1.0)

        index = []
        scale = [torch.zeros_like(value)]
        for step in range(self.length):
            entry = value // self.prod[step]
            value = value % self.prod[step]
            index.append(entry)
            scale.append(entry * self.prod[step])
        scale = scale[:-1]

        index = torch.stack(index, dim=-2)
        scale = torch.stack(scale, dim=-2)

        state = tuple(
            cspace.torch.classes.JointStateCollection.apply(
                self.spec,
                self.joint,
                torch.select(scale, dim=-2, index=step),
                min=0.0,
                max=1.0,
            )
            for step in range(self.length)
        )
        index = index.to(torch.int64)
        return state, index

    @functools.cached_property
    def prod(self):
        return torch.cumprod(
            torch.tensor(1.0 / self.bucket, dtype=torch.float64).expand([self.length]),
            dim=0,
        )


class MaskDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(self, /, length, entries, index):
        def f_data(entry, length):
            return torch.concatenate(
                (
                    entry,
                    torch.zeros(
                        (entry.shape[0], length - entry.shape[1], entry.shape[2])
                    ),
                ),
                dim=-2,
            )

        def f_mask(entry, length):
            return torch.concatenate(
                (
                    torch.ones((entry.shape[0], entry.shape[1])),
                    torch.zeros((entry.shape[0], length - entry.shape[1])),
                ),
                dim=-1,
            )

        def f_true(entry, length, index):
            return torch.select(index, dim=-2, index=entry.shape[1] - 1)

        self.data = torch.concatenate(
            list(f_data(entry, length) for entry in entries), dim=0
        )

        self.mask = torch.concatenate(
            list(f_mask(entry, length) for entry in entries), dim=0
        )

        self.true = torch.concatenate(
            list(f_true(entry, length, index) for entry in entries), dim=0
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key):
        return (self.head(key), self.data[key], self.mask[key], self.true[key])

    @abc.abstractmethod
    def head(self, key):
        raise NotImplementedError


class InverseDataset(MaskDataset):
    def __init__(
        self, /, spec, joint, link, base, bucket, length, e_data, total, noise=None
    ):
        total = total if noise is None else (noise * total)

        index = torch.randint(
            low=0, high=bucket, size=(total, length, len(joint)), dtype=torch.int64
        )
        if not noise:
            noise = torch.zeros((total, len(link), 6), dtype=torch.float64)
        else:
            std = torch.tensor(0.01, dtype=torch.float64).expand(total, len(link), 6)
            mean = torch.tensor(0.0, dtype=torch.float64).expand(total, len(link), 6)
            noise = torch.normal(mean, std)

        prod = torch.cumprod(
            torch.tensor(1.0 / bucket, dtype=torch.float64).expand([1, length, 1]),
            dim=-2,
        )

        scale = torch.multiply(index, prod)

        scale = torch.concatenate(
            (
                torch.zeros((total, 1, len(joint)), dtype=torch.float64),
                scale,
            ),
            dim=-2,
        )

        scale = torch.cumsum(scale, dim=-2)

        scale = scale % 1.0  # (0.0, 1.0)

        start = torch.rand(total, len(joint), dtype=scale.dtype, device=scale.device)

        state = tuple(
            cspace.torch.classes.JointStateCollection.apply(
                spec,
                joint,
                torch.select(scale, dim=-2, index=step),
                min=0.0,
                max=1.0,
            )
            for step in range(length)
        )

        pose = cspace.torch.classes.JointStateCollection.apply(
            spec,
            joint,
            torch.select(scale, dim=-2, index=length),
            min=0.0,
            max=1.0,
        ).forward(spec, *link, base=base)

        entries = list(
            e_data(state[0 : step + 1], pose, noise) for step in range(length)
        )
        super().__init__(length=length, entries=entries, index=index)

        start = e_data(
            [
                cspace.torch.classes.JointStateCollection.apply(
                    spec,
                    joint,
                    start,
                    min=0.0,
                    max=1.0,
                )
            ],
            pose,
            None,
        )

        self.start = torch.concatenate(list(start for step in range(length)), dim=0)

    def head(self, key):
        return self.start[key]


class InverseKinematics(cspace.torch.classes.InverseKinematics, JointStateEncoding):
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

    def inverse(self, pose, state, repeat=None):
        def f_encode(pose, zero, processed):
            return [zero] + processed, pose, zero  # start

        def f_decode(pred, zero, processed, choice):
            return [zero] + processed, pred, choice

        def f_selection(final, transform):
            position, orientation = self.forward(final).data

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

            selection = torch.min(loss, dim=0)
            selection = torch.reshape(selection.indices, [-1])

            position = torch.select(final.data, dim=0, index=selection)

            return cspace.torch.classes.JointStateCollection(
                self.spec, final.name, position
            )

        def f_choice(pred):
            pred = torch.transpose(pred, -1, -2)

            categorical = torch.distributions.categorical.Categorical(logits=pred)

            return categorical.sample()

        choice = f_choice if repeat else functools.partial(torch.argmax, dim=-2)

        self.model.eval()
        with torch.no_grad():
            if repeat:
                position, orientation = pose.data
                position = position.unsqueeze(0).expand(
                    *(tuple([repeat]) + pose.batch + (3, len(pose.name)))
                )
                orientation = orientation.unsqueeze(0).expand(
                    *(tuple([repeat]) + pose.batch + (4, len(pose.name)))
                )
                pose = cspace.torch.classes.LinkPoseCollection(
                    pose.base, pose.name, position, orientation
                )

                position = state.data
                position = position.unsqueeze(0).expand(
                    *(tuple([repeat]) + state.batch + tuple([len(state.name)]))
                )
                state = cspace.torch.classes.JointStateCollection(
                    self.spec, state.name, position
                )

                position, orientation = pose.data
                transform = cspace.torch.classes.Transform(
                    xyz=torch.transpose(position, -1, -2),
                    rot=cspace.torch.ops.qua_to_rot(
                        torch.transpose(orientation, -1, -2)
                    ),
                )

            zero = cspace.torch.classes.JointStateCollection.apply(
                self.spec,
                self.joint,
                torch.zeros_like(state.data),
                min=0.0,
                max=1.0,
            )

            processed = []
            for step in range(self.length):
                data, mask = self.encode(*f_encode(pose, zero, processed))

                pred = self.model(data, mask)

                processed.append(self.decode(*f_decode(pred, zero, processed, choice)))

            return f_selection(processed[-1], transform) if repeat else processed[-1]

    def train(
        self,
        *,
        logger,
        accelerator,
        total,
        noise=None,
        load=None,
        save=None,
        batch=None,
        start=None,
        limit=None,
    ):
        batch = batch if batch is not None else 128
        start = start if start is not None else 0

        logger.info(
            "[Train] ----- Dataset: (batch={}, total={}, noise={}) - (start={}, limit={}) - creation".format(
                batch, total, noise, start, limit
            )
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )
        model, optimizer, scheduler = accelerator.prepare(
            self.model, optimizer, scheduler
        )
        accelerator.load_state(load) if load else None

        model.train()
        for epoch in itertools.count(start) if limit is None else range(start, limit):
            logger.info(
                "[Train] ----- Dataset: (batch={}, total={}, noise={}) - (epoch={}) - creation".format(
                    batch, total, noise, epoch
                )
            )
            dataset = cspace.transformers.InverseDataset(
                spec=self.spec,
                joint=self.joint,
                link=self.link,
                base=self.base,
                bucket=self.bucket,
                length=self.length,
                e_data=self.e_data,
                total=total,
                noise=noise,
            )

            assert len(dataset) == self.length * total * (noise if noise else 1)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch,
                shuffle=True,
            )
            dataloader = accelerator.prepare(dataloader)

            loss_count, loss_total = 0, 0
            for index, (head, data, mask, true) in enumerate(dataloader):
                data, mask = self.e_compose(head, data, mask)
                pred = model(data, mask)
                loss = self.loss_fn(
                    torch.unflatten(pred, -1, (self.bucket, -1)),
                    true,
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
                    "[Train] ----- Dataset: (batch={}, total={}, noise={}) - (epoch={}) - {}/{} - Loss: {}".format(
                        batch,
                        total,
                        noise,
                        epoch,
                        loss_count,
                        total * (noise if noise else 1) * self.length,
                        loss_total / loss_count,
                    )
                )

            accelerator.save_state(save) if save else None
            (
                accelerator.save(self, pathlib.Path(save).joinpath("kinematics.pth"))
                if save
                else None
            )
            logger.info(
                "[Train] ----- Dataset: (batch={}, total={}, batch={}) - (epoch={}) - complete".format(
                    batch, total, noise, epoch
                )
            )

        logger.info(
            "[Train] ----- Dataset: (batch={}, total={}, noise={}) - (start={}, limit={}) - complete".format(
                batch, total, noise, start, limit
            )
        )

    def encode(self, state, pose, start):
        data = self.e_data(state, pose, noise=None)
        mask = torch.ones(data.shape[:-1])
        head = self.e_data([start], pose, noise=None)

        return self.e_compose(head, data, mask)

    def e_compose(self, head, data, mask):
        # mask = torch.concatenate(
        #    (
        #        torch.ones(
        #            mask.shape[:-1] + tuple([head.shape[-2]]),
        #            dtype=mask.dtype,
        #            device=mask.device,
        #        ),
        #        mask,
        #    ),
        #    dim=-1,
        # )
        # data = torch.concatenate((head, data), dim=-2)

        return data, mask

    def e_data(self, state, pose, noise):
        def f_value(self, entry, pose, noise):
            def f_pose(mark, pose, noise, index, name):
                entry = mark.transform(name).inverse() * pose.transform(name)
                if noise is not None:
                    xyz, rot = cspace.torch.ops.se3_exp(
                        torch.select(noise, dim=-2, index=index)
                    )
                    entry = entry * cspace.torch.classes.Transform(xyz=xyz, rot=rot)
                return entry.log

            assert entry.batch == pose.batch, "{} vs. {}".format(
                entry.batch, pose.batch
            )

            mark = self.forward(entry)
            value = [entry.data] + list(
                f_pose(mark, pose, noise, index, name)
                for index, name in enumerate(pose.name)
            )

            value = torch.concatenate(value, dim=-1)

            value = torch.reshape(value, pose.batch + (1, -1))

            return value

        value = tuple(f_value(self, entry, pose, noise) for entry in state)

        return torch.concatenate(value, dim=-2)


class PerceptionDataset(MaskDataset):
    def __init__(
        self, /, spec, joint, length, bucketize, function, e_data, image, label
    ):
        def f(entry):
            entries = list(
                pathlib.Path(image).glob(
                    str(entry.relative_to(label)).removesuffix(".json") + ".*"
                )
            )
            entries = list(
                (file, entry)
                for file in entries
                if file.suffix in (".png", ".jpg", ".jpeg", ".bmp")
            )
            return next(iter(entries), None)

        def f_value(entry):
            with open(entry) as f:
                entry = json.load(f)
            entry = torch.tensor(
                tuple(entry[name] for name in joint), dtype=torch.float64
            )
            return entry

        entries = list(f(entry) for entry in pathlib.Path(label).rglob("*.json"))
        entries = list(entry for entry in entries if entry)
        image, label = list(zip(*entries))

        value = torch.stack(list(f_value(entry) for entry in label), dim=0)

        state, index = bucketize(
            cspace.torch.classes.JointStateCollection(spec, joint, value)
        )

        entries = list(e_data(state[0 : step + 1]) for step in range(length))
        super().__init__(length=length, entries=entries, index=index)

        self.image = list(
            itertools.chain.from_iterable(list(image for step in range(length)))
        )

        self.function = function

    def head(self, key):
        return torch.squeeze(self.function(self.image[key]), dim=0)


class PerceptionKinematics(
    cspace.torch.classes.PerceptionKinematics, JointStateEncoding
):
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(
        self, description, base=None, model=None, vision=None, bucket=None, length=None
    ):
        super().__init__(description, base=base)
        if model:
            self.bucket = bucket if bucket else 10
            self.length = length if length else 3

            self.vision = transformers.AutoModel.from_pretrained(vision)
            for param in self.vision.parameters():
                param.requires_grad = False
            self.processor = transformers.AutoImageProcessor.from_pretrained(vision)

            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)
            input_embeddings = torch.nn.Linear(
                (len(self.joint) * 1 + self.vision.config.hidden_size),
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

    def perception(self, observation):
        def f_encode(pose, zero, processed):
            return [zero] + processed, pose, zero  # start

        def f_decode(pred, zero, processed, choice):
            return [zero] + processed, pred, choice

        choice = functools.partial(torch.argmax, dim=-2)

        self.model.eval()
        with torch.no_grad():
            batch = observation.shape[:-3]

            pose = self.vision.to(observation.device)(
                pixel_values=torch.reshape(
                    observation, tuple([-1]) + observation.shape[-3:]
                )
            ).last_hidden_state
            pose = torch.reshape(pose, batch + pose.shape[-2:])

            zero = cspace.torch.classes.JointStateCollection.apply(
                self.spec,
                self.joint,
                torch.zeros(
                    batch + tuple([len(self.joint)]), device=observation.device
                ),
                min=0.0,
                max=1.0,
            )

            processed = []
            for step in range(self.length):
                data, mask = self.encode(*f_encode(pose, zero, processed))

                pred = self.model(data, mask)

                processed.append(self.decode(*f_decode(pred, zero, processed, choice)))

            return processed[-1]

    def image(self, file, device=None):
        return torch.as_tensor(
            self.processor(
                PIL.Image.open(
                    io.BytesIO(file) if isinstance(file, bytes) else file
                ).convert("RGB"),
                return_tensors="pt",
            ).pixel_values[0],
            device=device,
        )

    def train(
        self,
        *,
        logger,
        accelerator,
        image,
        label,
        load=None,
        save=None,
        batch=None,
        start=None,
        limit=None,
    ):
        batch = batch if batch is not None else 128
        start = start if start is not None else 0

        logger.info(
            "[Train] ----- Dataset: (batch={}, image={}, label={}) - (start={}, limit={}) - creation".format(
                batch, image, label, start, limit
            )
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )
        dataset = cspace.transformers.PerceptionDataset(
            spec=self.spec,
            joint=self.joint,
            length=self.length,
            bucketize=self.bucketize,
            function=self.image,
            e_data=self.e_data,
            image=image,
            label=label,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch,
            shuffle=True,
        )
        model, optimizer, scheduler, dataloader, vision = accelerator.prepare(
            self.model, optimizer, scheduler, dataloader, self.vision
        )
        accelerator.load_state(load) if load else None

        logger.info(
            "[Train] ----- Dataset: (batch={}, image={}, label={}) - (start={}, limit={}) - (total={})".format(
                batch, image, label, start, limit, len(dataset)
            )
        )

        model.train(), vision.eval()
        for epoch in itertools.count(start) if limit is None else range(start, limit):
            logger.info(
                "[Train] ----- Dataset: (batch={}, image={}, label={}) - (epoch={}) - creation".format(
                    batch, image, label, epoch
                )
            )

            loss_total, loss_count = 0, 0
            for index, (head, data, mask, true) in enumerate(dataloader):
                with torch.no_grad():
                    head = vision(pixel_values=head).last_hidden_state
                data, mask = self.e_compose(head, data, mask)
                pred = model(data, mask)
                loss = self.loss_fn(
                    torch.unflatten(pred, -1, (self.bucket, -1)),
                    true,
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
                    "[Train] ----- Dataset: (batch={}, imagel={}, label={}) - (epoch={}) - {}/{} - Loss: {}".format(
                        batch,
                        image,
                        label,
                        epoch,
                        loss_count,
                        len(dataset) * self.length,
                        loss_total / loss_count,
                    )
                )
            accelerator.save_state(save) if save else None
            (
                accelerator.save(self, pathlib.Path(save).joinpath("kinematics.pth"))
                if save
                else None
            )
            logger.info(
                "[Train] ----- Dataset: (batch={}, image={}, label={}) - (epoch={}) - complete".format(
                    batch, image, label, epoch
                )
            )

        logger.info(
            "[Train] ----- Dataset: (batch={}, image={}, label={}) - (start={}, limit={}) - complete".format(
                batch, image, label, start, limit
            )
        )

    def encode(self, state, pose, start):
        data = self.e_data(state)
        mask = torch.ones(data.shape[:-1])
        head = pose
        return self.e_compose(head, data, mask)

    def e_compose(self, head, data, mask):
        mask = torch.concatenate(
            (
                torch.ones(
                    mask.shape[:-1] + tuple([head.shape[-2]]),
                    dtype=mask.dtype,
                    device=mask.device,
                ),
                mask,
            ),
            dim=-1,
        )

        total = data.shape[-1] + head.shape[-1]

        head = torch.concatenate(
            (
                torch.zeros(
                    (head.shape[:-1] + tuple([total - head.shape[-1]])),
                    dtype=head.dtype,
                    device=head.device,
                ),
                head,
            ),
            dim=-1,
        )
        data = torch.concatenate(
            (
                data,
                torch.zeros(
                    (data.shape[:-1] + tuple([total - data.shape[-1]])),
                    dtype=data.dtype,
                    device=data.device,
                ),
            ),
            dim=-1,
        )

        data = torch.concatenate((head, data), dim=-2)

        return data, mask

    def e_data(self, state):
        value = torch.stack(tuple(entry.data for entry in state), dim=-2)

        return value
