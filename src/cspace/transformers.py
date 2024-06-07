import cspace.cspace.classes
import cspace.torch.classes
import transformers
import dataclasses
import functools
import itertools
import logging
import torch
import abc


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataIndex:
    e: str | None
    name: str | None
    field: int | None


class DataBlock(abc.ABC):
    def __init__(self, dimension):
        self.dimension = int(dimension)

    def encode(self, data):
        value = torch.as_tensor(data, dtype=torch.float64)
        value = torch.special.expit(value)
        value = torch.clip(
            (value * (self.dimension - 1)).to(torch.int64),
            min=0,
            max=(self.dimension - 1),
        )
        return value

    def decode(self, data):
        value = torch.as_tensor(data, dtype=torch.int64)
        value = torch.clip(
            (value % self.dimension).to(torch.float64) / (self.dimension - 1),
            min=0.0,
            max=1.0,
        )
        value = torch.special.logit(value)
        return value


@dataclasses.dataclass(frozen=True, init=False)
class NoneIndex(DataIndex):
    def __init__(self):
        object.__setattr__(self, "e", None)
        object.__setattr__(self, "name", None)
        object.__setattr__(self, "field", None)


@dataclasses.dataclass(frozen=True, init=False)
class PoseIndex(DataIndex):
    def __init__(self, name, field):
        object.__setattr__(self, "e", "pose")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "field", field)


@dataclasses.dataclass(frozen=True, init=False)
class JointIndex(DataIndex):
    def __init__(self, name):
        object.__setattr__(self, "e", "joint")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "field", None)


class BlockEncoding(abc.ABC):
    def __init__(self, spec, link):
        self.blocks = []
        self.blocks.append((NoneIndex(), DataBlock(dimension=1)))
        for name in link:
            for field in range(6):
                self.blocks.append(
                    (PoseIndex(name=name, field=field), DataBlock(dimension=1000))
                )
        for name in tuple(joint.name for joint in spec.joint if joint.motion.call):
            self.blocks.append((JointIndex(name=name), DataBlock(dimension=1000)))

    def encode(self, data):
        index, value = data
        index = self.indices.index(index)
        count = sum(map(lambda i: self.entries[i].dimension, range(index)))
        block = self.entries[index]
        return count + block.encode(value)

    def decode(self, data):
        count = 0
        for index, block in self.blocks:
            if count <= data and data < count + block.dimension:
                return index, block.decode(data - count)
            count += block.dimension
        raise ValueError(f"{data}")

    @functools.cached_property
    def indices(self):
        return tuple(index for index, block in self.blocks)

    @functools.cached_property
    def entries(self):
        return tuple(block for index, block in self.blocks)

    @functools.cached_property
    def dimension(self):
        return sum(map(lambda block: block.dimension, self.entries))


class Model(torch.nn.Module):
    def __init__(self, transformer, dimension):
        super().__init__()
        self.transformer = transformer
        self.embedding = torch.nn.Embedding(
            dimension, transformer.get_input_embeddings().embedding_dim
        )
        transformer.get_input_embeddings().reset_parameters()
        for param in transformer.get_input_embeddings().parameters():
            param.requires_grad = False
        transformer.get_output_embeddings().reset_parameters()
        for param in transformer.get_output_embeddings().parameters():
            param.requires_grad = False

    def forward(self, batch):
        entries = [entry for entry in batch]

        device = self.transformer.get_input_embeddings().weight.device

        data = entries
        mask = list(map(lambda e: torch.ones(len(e)), data))

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).to(device)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(device)
        data = self.transformer(
            inputs_embeds=self.embedding(data),
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        data = torch.stack(
            [data[i, j] for i, j in enumerate(torch.count_nonzero(mask, dim=1) - 1)]
        )

        data = torch.mm(data, self.embedding.weight.T).cpu()
        return data


class Kinematics:
    spec: cspace.cspace.classes.Spec
    base: str
    link: tuple[str]
    model: torch.nn.Module

    def __init__(self, description, *link, base=None, model=None):
        spec = cspace.cspace.classes.Spec(description=description)
        assert (not base) or (base in spec.link)
        base = str(base) if base else spec.base
        assert (not link) or all([(item in spec.link) for item in link])
        link = tuple(link) if link else spec.link

        self.spec = spec
        self.base = base
        self.link = link
        self.model = Model(
            transformer=transformers.AutoModelForCausalLM.from_pretrained(model),
            dimension=self.encoding.dimension,
        )

    def forward(self, state):
        return state.forward(self.spec, *self.link, base=self.base)

    def inverse(self, pose):
        data = self.tokenize(pose)
        name = list(joint.name for joint in self.spec.joint if joint.motion.call)
        with torch.no_grad():
            for index in range(len(name)):
                pred = self.model(torch.tensor(data).unsqueeze(0))
                pred = torch.argmax(pred)
                data = data + tuple([pred])
        data = data[:-1] + tuple([0])
        return self.assembly(data)

    def tokenize(self, data):
        def f_pose(spec, link, base, zero, link_transforms):
            zero_transform = zero.transform(spec, link, base)
            link_transform = link_transforms[link]
            transform = zero_transform.inverse() * link_transform
            return [
                self.encoding.encode((PoseIndex(name=link, field=field), entry))
                for field, entry in enumerate(
                    cspace.torch.ops.se3_log(transform.xyz, transform.rot)
                )
            ]

        name = tuple(joint.name for joint in self.spec.joint if joint.motion.call)
        zero = cspace.torch.classes.JointStateCollection(name, tuple(0.0 for e in name))

        if isinstance(data, cspace.cspace.classes.JointStateCollection):
            assert tuple(sorted(data.name)) == tuple(sorted(name))
            link_transforms = {
                link: data.transform(self.spec, link, self.base) for link in self.link
            }
            state = data
        elif isinstance(data, cspace.cspace.classes.LinkPoseCollection):

            def f_link(pose, base):
                assert pose.base == base
                transform = cspace.torch.classes.Transform(
                    xyz=pose.position, rot=cspace.torch.ops.qua_to_rot(pose.orientation)
                )
                return transform

            assert tuple(sorted(data.name)) == tuple(sorted(self.link))
            link_transforms = {
                link: f_link(data(link), self.base) for link in self.link
            }
            state = None
        else:
            raise ValueError(f"{data}")

        entries = tuple(
            itertools.chain.from_iterable(
                tuple(
                    f_pose(self.spec, link, self.base, zero, link_transforms)
                    for link in self.link
                )
            )
        )
        entries = entries + tuple([self.encoding.encode((NoneIndex(), 0.0))])

        if state is not None:
            entries = entries + tuple(
                self.encoding.encode((JointIndex(name=name), state(name).position))
                for name in state.name
            )
            entries = entries + tuple([self.encoding.encode((NoneIndex(), 0.0))])
        return entries

    def assembly(self, data):
        entries = tuple(map(self.encoding.decode, data))
        index = list(i for i, (index, entry) in enumerate(entries) if index.e is None)
        index = next(iter(index))
        poses = entries[:index]
        entries = entries[index + 1 :]
        index = list(i for i, (index, entry) in enumerate(entries) if index.e is None)
        if len(index) == 0:
            raise NotImplementedError

        states = entries
        name = tuple(joint.name for joint in self.spec.joint if joint.motion.call)
        position = [0.0 for e in name]
        for index, entry in states:
            if index.e == "joint":
                position[name.index(index.name)] += entry
        return cspace.torch.classes.JointStateCollection(name, position)

    @property
    def encoding(self):
        return BlockEncoding(self.spec, self.link)

    def train(self, total=None, epoch=None, batch=None, device=None, seed=None):
        entry_total = total if total else 1024
        epoch_total = epoch if epoch else 1
        batch_size = batch if batch else 128

        generator = torch.Generator().manual_seed(seed)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters())

        dataset = []

        entry_index = 0
        while entry_index < entry_total:
            name = tuple(joint.name for joint in self.spec.joint if joint.motion.call)
            position = torch.rand(len(name), generator=generator, dtype=torch.float64)
            state = cspace.torch.classes.JointStateCollection(name, position)
            encoded = self.tokenize(state)
            offset = encoded.index(0) + 1
            for length in range(offset, len(encoded)):
                dataset.append(torch.tensor(encoded[: length + 1]))
                entry_index += 1
            if entry_index % 512 == 0:
                logging.getLogger(__name__).info(f"Dataset[{entry_index}]")
        dataset = dataset[:entry_total]
        logging.getLogger(__name__).info(f"Dataset[{entry_total}]")

        input_dataset = list(map(lambda e: e[:-1], dataset))
        label_dataset = list(map(lambda e: e[-1], dataset))

        input_loader = torch.utils.data.DataLoader(
            input_dataset,
            batch_size=batch_size,
            collate_fn=list,
        )

        label_loader = torch.utils.data.DataLoader(
            label_dataset,
            batch_size=batch_size,
        )

        self.model.train()
        self.model.to(device)
        for epoch in range(epoch_total):
            total, count = 0, 0
            for batch, (data, true) in enumerate(zip(input_loader, label_loader)):
                pred = self.model(data)
                loss = loss_fn(pred, true)
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
