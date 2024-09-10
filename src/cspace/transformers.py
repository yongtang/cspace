import cspace.cspace.classes
import cspace.torch.classes
import transformers
import PIL.Image
import functools
import pathlib
import torch
import json
import io


class Model(torch.nn.Module):
    def __init__(self, transformer, input_embeddings, output_embeddings):
        super().__init__()
        transformer.get_input_embeddings().reset_parameters()
        for param in transformer.get_input_embeddings().parameters():
            param.requires_grad = False
        if transformer.get_output_embeddings():
            transformer.get_output_embeddings().reset_parameters()
            for param in transformer.get_output_embeddings().parameters():
                param.requires_grad = False

        self.transformer = transformer
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings

    def batch(self, data, length):
        data = torch.reshape(data, [-1] + list(data.shape[-2:]))

        mask = torch.concatenate(
            (
                torch.ones((data.shape[0], data.shape[1])),
                torch.zeros((data.shape[0], length - data.shape[1])),
            ),
            dim=-1,
        )

        data = torch.concatenate(
            (
                data,
                torch.zeros(
                    (data.shape[0], length - data.shape[1], data.shape[2]),
                    dtype=data.dtype,
                    device=data.device,
                ),
            ),
            dim=-2,
        )

        return data, mask

    def forward(self, data, mask):
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
        return data


class JointStateEncoding:
    def decode(self, state, pred):
        pred = torch.unflatten(pred, -1, (self.bucket, -1))
        assert pred.shape[-2:] == (self.bucket, len(self.joint))

        prod = self.prod[len(state) - 1]
        index = torch.argmax(pred, dim=-2)

        scale = state[-1].scale(self.spec, min=0.0, max=1.0)
        scale = scale + torch.multiply(index, prod)
        scale = scale % 1.0

        state = cspace.torch.classes.JointStateCollection.apply(
            self.spec, self.joint, scale, min=0.0, max=1.0
        )
        return state

    def bucketize(self, state):
        value = state.scale(spec=self.spec, min=0.0, max=1.0)

        true = []
        scale = [torch.zeros_like(value)]
        for step in range(self.length):
            entry = value // self.prod[step]
            value = value % self.prod[step]
            true.append(entry)
            scale.append(entry * self.prod[step])
        scale = scale[:-1]

        true = torch.stack(true, dim=-2)
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
        true = true.to(torch.int64)
        return state, true

    @functools.cached_property
    def prod(self):
        return torch.cumprod(
            torch.tensor(1.0 / self.bucket, dtype=torch.float64).expand([self.length]),
            dim=0,
        )


class InverseDataset(torch.utils.data.Dataset):
    def __init__(
        self, *, spec, joint, link, base, bucket, length, total, noise, encode, device
    ):
        total = total if noise is None else (noise * total)

        index = torch.randint(
            low=0,
            high=bucket,
            size=(total, length, len(joint)),
            dtype=torch.int64,
            device=device,
        )
        if not noise:
            noise = torch.zeros(
                (total, len(link), 6), dtype=torch.float64, device=device
            )
        else:
            std = torch.tensor(0.01, dtype=torch.float64, device=device).expand(
                total, len(link), 6
            )
            mean = torch.tensor(0.0, dtype=torch.float64, device=device).expand(
                total, len(link), 6
            )
            noise = torch.normal(mean, std)

        prod = torch.cumprod(
            torch.tensor(1.0 / bucket, dtype=torch.float64, device=device).expand(
                [1, length, 1]
            ),
            dim=-2,
        )

        scale = torch.multiply(index, prod)

        scale = torch.concatenate(
            (
                torch.rand(total, 1, len(joint), dtype=torch.float64, device=device),
                scale,
            ),
            dim=-2,
        )

        scale = torch.cumsum(scale, dim=-2)

        scale = scale % 2.0  # (0.0, 1.0)

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
            encode(state[0 : step + 1], pose, noise) for step in range(length)
        )

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
        return (self.data[key], self.mask[key], self.true[key])


class InverseKinematics(cspace.torch.classes.InverseKinematics, JointStateEncoding):
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(
        self, description, *link, base=None, model=None, bucket=None, length=None
    ):
        super().__init__(description, *link, base=base)
        if model:
            self.bucket = bucket if bucket else 10
            self.length = length if length else 3

            assert model == "gpt2"
            config = transformers.GPT2Config(
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                summary_first_dropout=0,
            )
            transformer = transformers.AutoModel.from_config(config)

            input_embeddings = torch.nn.Linear(
                (len(self.joint) * 1 + len(self.link) * 6),
                transformer.get_input_embeddings().embedding_dim,
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )
            output_embeddings = torch.nn.Linear(
                transformer.get_input_embeddings().embedding_dim,
                (len(self.joint) * self.bucket),
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )

            self.model = Model(transformer, input_embeddings, output_embeddings)

    def inverse(self, pose, state, repeat=None):

        def f_pose(pose, count):
            position, orientation = pose.data

            position = position.unsqueeze(0).expand(
                *(tuple([count]) + pose.batch + (3, len(pose.name)))
            )
            orientation = orientation.unsqueeze(0).expand(
                *(tuple([count]) + pose.batch + (4, len(pose.name)))
            )

            return cspace.torch.classes.LinkPoseCollection(
                pose.base, pose.name, position, orientation
            )

        def f_state(state, count):
            position = state.data
            position = position.unsqueeze(0).expand(
                *(tuple([count]) + state.batch + tuple([len(state.name)]))
            )
            return cspace.torch.classes.JointStateCollection(
                self.spec, state.name, position
            )

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

        def f_encode(pose, state, processed, repeat):
            return [f_state(state, repeat)] + processed, f_pose(pose, repeat)

        def f_decode(pred, state, processed, repeat):
            return [f_state(state, repeat)] + processed, pred

        repeat = repeat if repeat else 16

        with torch.no_grad():
            position, orientation = pose.data
            transform = cspace.torch.classes.Transform(
                xyz=torch.transpose(position, -1, -2),
                rot=cspace.torch.ops.qua_to_rot(torch.transpose(orientation, -1, -2)),
            )

            processed = []
            for step in range(self.length):
                data = self.encode(*f_encode(pose, state, processed, repeat))

                pred = self.model(*self.model.batch(data, self.length))

                pred = torch.reshape(pred, data.shape[:-2] + pred.shape[-1:])

                processed.append(self.decode(*f_decode(pred, state, processed, repeat)))
            return f_selection(processed[-1], transform)

    def train(
        self,
        *,
        logger,
        accelerator,
        save=None,
        batch=None,
        total=None,
        lr=None,
        noise=None,
    ):
        batch = batch if batch is not None else 128
        total = total if total is not None else 1024
        lr = lr if lr is not None else 1e-5

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )

        logger.info(
            "[Train] ----- Dataset: (length={}, total={}, noise={}) - (batch={}) - creation".format(
                self.length, total, noise, batch
            )
        )

        if total > 0:

            dataset = cspace.transformers.InverseDataset(
                spec=self.spec,
                joint=self.joint,
                link=self.link,
                base=self.base,
                bucket=self.bucket,
                length=self.length,
                total=total,
                noise=noise,
                encode=self.encode,
                device=None,
            )
            assert len(dataset) == self.length * total * (noise if noise else 1)
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
            for index, (data, mask, true) in enumerate(dataloader):
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
                    "[Train] ----- Dataset: (length={}, total={}, noise={}) - (batch={}) - (count={}/{}) - Loss: {}".format(
                        self.length,
                        total,
                        noise,
                        batch,
                        (index + 1) * batch,
                        len(dataset),
                        loss_total / loss_count,
                    )
                )
        accelerator.save(self, save) if save else None
        logger.info(
            "[Train] ----- Dataset: (length={}, total={}, noise={}) - (batch={}) - complete".format(
                self.length, total, noise, batch
            )
        )

    def encode(self, state, pose, noise=None):
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


class PerceptionDataset(torch.utils.data.Dataset):
    def __init__(self, /, image, label, spec, joint, function, bucketize):
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

        def f_label(file):
            with open(file) as f:
                entry = json.load(f)
                entry = torch.tensor(
                    tuple(entry[name] for name in joint), dtype=torch.float64
                )
                return entry

        entries = list(filter(None, map(f, pathlib.Path(label).rglob("*.json"))))

        image, label = list(zip(*entries))  # unzip

        position = torch.stack(list(f_label(file) for file in label))

        state = cspace.torch.classes.JointStateCollection(spec, joint, position)

        state, true = bucketize(state)

        value = torch.stack(list(entry.data for entry in state), dim=-2)

        self.function = function
        self.image = image
        self.value = value
        self.true = true

    def __len__(self):
        return len(self.image)

    def __getitem__(self, key):
        observation = torch.squeeze(self.function(self.image[key]), dim=0)
        value = self.value[key]
        true = self.true[key]

        return observation, value, true


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
            return [zero] + processed, pose

        def f_decode(pred, zero, processed):
            return [zero] + processed, pred

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
                data = self.encode(*f_encode(pose, zero, processed))

                pred = self.model(*self.model.batch(data, pose.shape[-2] + self.length))

                pred = torch.reshape(pred, data.shape[:-2] + pred.shape[-1:])

                processed.append(self.decode(*f_decode(pred, zero, processed)))

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
        save=None,
        batch=None,
        lr=None,
    ):
        batch = batch if batch is not None else 128
        lr = lr if lr is not None else 1e-5

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )

        logger.info(
            "[Train] ----- Dataset: (image={}, label={}, batch={}) - creation".format(
                image, label, batch
            )
        )

        if image:
            dataset = cspace.transformers.PerceptionDataset(
                image=image,
                label=label,
                spec=self.spec,
                joint=self.joint,
                function=self.image,
                bucketize=self.bucketize,
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

            logger.info(
                "[Train] ----- Dataset: (total={}, batch={})".format(
                    len(dataset), batch
                )
            )

            vision = accelerator.prepare(self.vision)

            loss_total, loss_count = 0, 0
            for index, (observation, value, true) in enumerate(dataloader):
                state = list(
                    cspace.torch.classes.JointStateCollection(
                        self.spec, self.joint, torch.select(value, dim=-2, index=index)
                    )
                    for index in range(self.length)
                )
                with torch.no_grad():
                    pose = vision(pixel_values=observation).last_hidden_state

                for step in range(self.length):
                    data = self.encode(state[0 : step + 1], pose)
                    pred = self.model(
                        *self.model.batch(data, pose.shape[-2] + self.length)
                    )
                    pred = torch.reshape(pred, data.shape[:-2] + pred.shape[-1:])
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
                    "[Train] ----- Dataset: (total={}, batch={}) - {}/{} - Loss: {}".format(
                        len(dataset),
                        batch,
                        (index + 1) * batch,
                        len(dataset),
                        loss_total / loss_count,
                    )
                )
        accelerator.save(self, save) if save else None
        logger.info(
            "[Train] ----- Dataset: (image={}, label={}, batch={}) - complete".format(
                image, label, batch
            )
        )

    def encode(self, state, pose):
        data = torch.stack(tuple(entry.data for entry in state), dim=-2)

        assert data.shape[:-2] == pose.shape[:-2]

        fill = data.shape[-1] + pose.shape[-1]

        data = torch.concatenate(
            (
                data,
                torch.zeros(
                    (data.shape[:-1] + tuple([fill - data.shape[-1]])),
                    dtype=data.dtype,
                    device=data.device,
                ),
            ),
            dim=-1,
        )
        pose = torch.concatenate(
            (
                torch.zeros(
                    (pose.shape[:-1] + tuple([fill - pose.shape[-1]])),
                    dtype=pose.dtype,
                    device=pose.device,
                ),
                pose,
            ),
            dim=-1,
        )

        return torch.concatenate((pose, data), dim=-2)
