import cspace.cspace.classes
import cspace.torch.classes
import transformers
import itertools
import logging
import torch


class Model(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.embedding = torch.nn.LazyLinear(
            transformer.get_input_embeddings().embedding_dim,
            device=transformer.get_input_embeddings().weight.device,
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
        entries = [entry for entry in batch]

        data = entries
        mask = list(map(lambda e: torch.ones(len(e)), data))

        data = (
            torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
            .to(self.embedding.weight.device)
            .to(self.embedding.weight.dtype)
        )
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(
            self.embedding.weight.device
        )
        data = self.transformer(
            inputs_embeds=self.embedding(data),
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        data = torch.stack(
            [data[i, j] for i, j in enumerate(torch.count_nonzero(mask, dim=1) - 1)]
        )

        data = torch.mm(data, self.embedding.weight).cpu()
        return data


class Kinematics:
    spec: cspace.cspace.classes.Spec
    base: str
    link: tuple[str]
    joint: tuple[str]
    model: torch.nn.Module
    bucket: int = 1000
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self, description, *link, base=None, model=None):
        spec = cspace.cspace.classes.Spec(description=description)
        assert (not base) or (base in spec.link)
        base = str(base) if base else spec.base
        assert (not link) or all([(item in spec.link) for item in link])
        link = tuple(link) if link else spec.link

        joint = list([e for e, _ in spec.route(e, base)] for e in link)
        joint = list(set(itertools.chain.from_iterable(joint)))
        joint = tuple(e for e in sorted(joint) if spec.joint(e).motion.call)

        self.spec = spec
        self.base = base
        self.link = link
        self.joint = joint
        self.model = (
            Model(transformer=transformers.AutoModelForCausalLM.from_pretrained(model))
            if model
            else None
        )

    def forward(self, state):
        return state.forward(self.spec, *self.link, base=self.base)

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

            zero_state = cspace.torch.classes.JointStateCollection(
                self.joint,
                torch.stack(
                    tuple(
                        torch.tensor(self.spec.joint(name).motion.zero).expand(batch)
                        for name in self.joint
                    ),
                    dim=-1,
                ),
            )

            return cspace.torch.classes.JointStateCollection(
                self.joint,
                tuple(
                    zero_state(name).apply(self.spec, delta_value[index])
                    for index, name in enumerate(self.joint)
                ),
            )

    def loss(self, pred, true):
        batch = {pred.shape[:-1], true.position.shape[:-1]}
        assert len(batch) == 1
        batch = next(iter(batch))

        pred_value = torch.unflatten(pred, -1, (-1, self.bucket))
        assert pred_value.shape[-2:] == torch.Size(
            [6 * len(self.link) + 1 * len(self.joint), self.bucket]
        )
        pred_value = pred_value[..., 6 * len(self.link) :, :]
        pred_value = torch.transpose(pred_value, -1, -2)

        zero_state = cspace.torch.classes.JointStateCollection(
            self.joint,
            torch.stack(
                tuple(
                    torch.tensor(self.spec.joint(name).motion.zero).expand(batch)
                    for name in self.joint
                ),
                dim=-1,
            ),
        )
        true_state = cspace.torch.classes.JointStateCollection(
            self.joint,
            torch.stack(tuple(true(name).position for name in self.joint), dim=-1),
        )
        true_delta = torch.stack(
            tuple(
                zero_state(name).delta(self.spec, true_state(name))
                for name in self.joint
            ),
            dim=-1,
        )
        true_value = true_delta * (self.bucket - 1)
        true_value = torch.clip(true_value.to(torch.int64), min=0, max=self.bucket - 1)
        return self.loss_fn(pred_value, true_value)

    def head(self, pose):
        batch = list(pose.position.shape[:-2])

        zero = self.forward(
            cspace.torch.classes.JointStateCollection(
                self.joint,
                torch.stack(
                    tuple(
                        torch.tensor(self.spec.joint(name).motion.zero).expand(batch)
                        for name in self.joint
                    ),
                    dim=-1,
                ),
            )
        )
        delta = torch.stack(
            tuple(zero(name).delta(self.spec, pose(name)) for name in self.link), dim=-1
        )
        blank = torch.zeros(batch + [1, len(self.joint)])

        delta = torch.reshape(delta, batch + [-1])
        blank = torch.reshape(blank, batch + [-1])

        value = torch.concatenate((delta, blank), dim=-1)
        value = value * (self.bucket - 1)
        value = torch.clip(value.to(torch.int64), min=0, max=self.bucket - 1)
        encoded = torch.nn.functional.one_hot(value, self.bucket)
        encoded = torch.flatten(encoded, -2, -1)
        encoded = torch.unsqueeze(encoded, -2)
        return encoded

    def train(self, total=None, epoch=None, batch=None, device=None, seed=None):
        entry_total = total if total else 1024
        epoch_total = epoch if epoch else 1
        batch_size = batch if batch else 128

        generator = torch.Generator().manual_seed(seed)

        optimizer = torch.optim.AdamW(self.model.parameters())

        zero = cspace.torch.classes.JointStateCollection(
            self.joint, tuple(self.spec.joint(name).motion.zero for name in self.joint)
        )

        logging.getLogger(__name__).info(
            "\n[Train] ----- Dataset: {} progress".format(entry_total)
        )
        dataset = list(
            cspace.torch.classes.JointStateCollection(
                self.joint,
                tuple(
                    zero(name).apply(
                        self.spec,
                        torch.rand(1, generator=generator, dtype=torch.float64),
                    )
                    for index, name in enumerate(self.joint)
                ),
            )
            for _ in range(entry_total)
        )
        logging.getLogger(__name__).info(
            "\n[Train] ----- Dataset: {} complete".format(entry_total)
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=cspace.torch.classes.JointStateCollection.stack,
        )

        self.model.train()
        self.model.to(device)
        for epoch in range(epoch_total):
            total, count = 0, 0
            for batch, true in enumerate(loader):
                data = self.head(self.forward(true))
                pred = self.model(data)
                loss = self.loss(pred, true)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                count += len(pred)
                total += loss.item()
                logging.getLogger(__name__).info(
                    "\n[Train] ----- Epoch {} [{}/{}] - Loss: {} [/Train]".format(
                        epoch,
                        count,
                        len(dataset),
                        total / count,
                    )
                )
