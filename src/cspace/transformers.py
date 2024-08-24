import cspace.cspace.classes
import cspace.torch.classes
import transformers
import PIL.Image
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


class JointStateDecode:
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


class InverseKinematics(cspace.torch.classes.InverseKinematics, JointStateDecode):
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

    def inverse(self, pose, repeat=None):

        repeat = repeat if repeat else 16

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
            "[Train] ----- Dataset: (total={}, batch={}) - creation".format(
                total, batch
            )
        )

        if total > 0:

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
            for index, (scale, true, delta) in enumerate(dataloader):
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
                    "[Train] ----- Dataset: (total={}, batch={}) - {}/{} - Loss: {}".format(
                        total,
                        batch,
                        (index + 1) * batch,
                        total,
                        loss_total / loss_count,
                    )
                )
        accelerator.save(self, save) if save else None
        logger.info(
            "[Train] ----- Dataset: (total={}, batch={}) - complete".format(
                total, batch
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


class PerceptionDataset(torch.utils.data.Dataset):
    def __init__(self, /, image, label, function, joint):
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

        self.entries = list(filter(None, map(f, pathlib.Path(label).rglob("*.json"))))
        self.function = function
        self.joint = joint

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        image, label = self.entries[key]
        image = self.function(image)
        image = torch.squeeze(image, dim=0)

        with open(label) as f:
            label = json.load(f)
        label = torch.tensor(
            tuple(label[name] for name in self.joint), dtype=torch.float64
        )

        return image, label


class PerceptionKinematics(cspace.torch.classes.PerceptionKinematics, JointStateDecode):
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
        with torch.no_grad():
            batch = observation.shape[:-3]

            state = [
                cspace.torch.classes.JointStateCollection.apply(
                    self.spec,
                    self.joint,
                    torch.zeros(batch + tuple([len(self.joint)])),
                    min=-1.0,
                    max=1.0,
                )
            ]

            for step in range(self.length):
                data = self.encode(state, observation)

                pred = self.model(data)

                state.append(self.decode(state, pred))

            state = state[-1]

            return state

    def image(self, file):
        return self.processor(
            PIL.Image.open(
                io.BytesIO(file) if isinstance(file, bytes) else file
            ).convert("RGB"),
            return_tensors="pt",
        ).pixel_values

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
                function=self.image,
                joint=self.joint,
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

            prod = (
                torch.cumprod(
                    torch.tensor(1.0 / self.bucket, dtype=torch.float64).expand(
                        [self.length, 1]
                    ),
                    dim=-2,
                )
                * self.bucket
            )

            zero = prod.expand(self.length, self.bucket) * torch.linspace(
                start=-1.0 + 1.0 / self.bucket,
                end=1.0 - 1.0 / self.bucket,
                steps=self.bucket,
                dtype=torch.float64,
            ).expand(self.length, self.bucket)

            prod = prod.expand(self.length, self.bucket - 1) * torch.linspace(
                start=-1.0 + 2.0 / self.bucket,
                end=1.0 - 2.0 / self.bucket,
                steps=self.bucket - 1,
                dtype=torch.float64,
            ).expand(self.length, self.bucket - 1)

            loss_total, loss_count = 0, 0
            for index, (observation, value) in enumerate(dataloader):
                value = cspace.torch.classes.JointStateCollection(self.joint, value)
                value = value.scale(spec=self.spec, min=-1.0, max=1.0).cpu()

                true = []
                for step in range(self.length):
                    entry = torch.bucketize(value, prod[step])
                    value = value - zero[step][entry]
                    true.append(entry)

                scale = [torch.zeros_like(value)]
                for step in range(self.length - 1):
                    entry = scale[-1] + zero[step][true[step]]
                    scale.append(entry)

                true = torch.stack(true, dim=-2)
                scale = torch.stack(scale, dim=-2)

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

                for step in range(self.length):
                    data = self.encode(state[0 : step + 1], observation)
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

    def encode(self, state, observation):
        def f_entry(self, entry):
            value = tuple(
                torch.unsqueeze(entry.position(self.spec, name), -1)
                for name in entry.name
            )
            value = tuple(entry.to(next(iter(value)).device) for entry in value)
            value = torch.concatenate(value, dim=-1)

            value = torch.reshape(value, entry.batch + (1, -1))

            return value

        value = tuple(f_entry(self, entry) for entry in state)

        value = tuple(entry.to(next(iter(value)).device) for entry in value)

        value = torch.concatenate(value, dim=-2)

        with torch.no_grad():
            print("XXXX - ", observation.device)
            print("YYYY - ", self.vision.device)
            pixel = self.vision(pixel_values=observation).last_hidden_state

        total = value.shape[-1] + pixel.shape[-1]

        pixel = torch.concatenate(
            (
                torch.zeros(
                    (pixel.shape[:-1] + tuple([total - pixel.shape[-1]])),
                    dtype=pixel.dtype,
                    device=pixel.device,
                ),
                pixel,
            ),
            dim=-1,
        )
        value = torch.concatenate(
            (
                value,
                torch.zeros(
                    (value.shape[:-1] + tuple([total - value.shape[-1]])),
                    dtype=value.dtype,
                    device=value.device,
                ),
            ),
            dim=-1,
        )

        value = torch.concatenate((pixel, value), dim=-2)

        return value
