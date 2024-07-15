import cspace.cspace.classes
import cspace.torch.classes
import transformers
import accelerate
import torch


class Model(torch.nn.Module):
    def __init__(self, transformer, input_embeddings, output_embeddings):
        super().__init__()
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


class Kinematics(cspace.torch.classes.Kinematics):
    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self, description, *link, base=None, model=None, bucket=None):
        super().__init__(description, *link, base=base, model=model)
        if model:
            self.bucket = bucket if bucket else 1000
            transformer = transformers.AutoModelForCausalLM.from_pretrained(model)

            for param in transformer.get_input_embeddings().parameters():
                param.requires_grad = False
            transformer.get_output_embeddings().reset_parameters()
            for param in transformer.get_output_embeddings().parameters():
                param.requires_grad = False

            input_embeddings = torch.nn.Linear(
                (6 * len(self.link) + 1 * len(self.joint)) * self.bucket,
                transformer.get_input_embeddings().embedding_dim,
                dtype=transformer.get_input_embeddings().weight.dtype,
                bias=False,
            )
            output_embeddings = torch.nn.Linear(
                transformer.get_input_embeddings().embedding_dim,
                (6 * len(self.link) + 1 * len(self.joint)) * self.bucket,
                dtype=transformer.get_output_embeddings().weight.dtype,
                bias=False,
            )

            self.model = Model(transformer, input_embeddings, output_embeddings)

    def encode(self, pose):
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

    def decode(self, pred):
        batch = pred.shape[:-1]

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
            for batch, (pose, true) in enumerate(dataloader):
                data = self.encode(pose)
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
