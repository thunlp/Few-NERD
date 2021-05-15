import torch
import torch.nn as nn


START_ID = 0
O_ID = 1

class ViterbiDecoder:
    """
    Generalized Viterbi decoding
    """

    def __init__(self, n_tag, abstract_transitions, tau):
        """
        We assume the batch size is 1, so no need to worry about PAD for now
        n_tag: START, O, and I_Xs
        """
        super().__init__()
        self.transitions = self.project_target_transitions(n_tag, abstract_transitions, tau)

    @staticmethod
    def project_target_transitions(n_tag, abstract_transitions, tau):
        s_o, s_i, o_o, o_i, i_o, i_i, x_y = abstract_transitions
        # self transitions for I-X tags
        a = torch.eye(n_tag) * i_i
        # transitions from I-X to I-Y
        b = torch.ones(n_tag, n_tag) * x_y / (n_tag - 3)
        c = torch.eye(n_tag) * x_y / (n_tag - 3)
        transitions = a + b - c
        # transition from START to O
        transitions[START_ID, O_ID] = s_o
        # transitions from START to I-X
        transitions[START_ID, O_ID+1:] = s_i / (n_tag - 2)
        # transition from O to O
        transitions[O_ID, O_ID] = o_o
        # transitions from O to I-X
        transitions[O_ID, O_ID+1:] = o_i / (n_tag - 2)
        # transitions from I-X to O
        transitions[O_ID+1:, O_ID] = i_o
        # no transitions to START
        transitions[:, START_ID] = 0.

        powered = torch.pow(transitions, tau)
        summed = powered.sum(dim=1)

        transitions = powered / summed.view(n_tag, 1)

        transitions = torch.where(transitions > 0, transitions, torch.tensor(.000001))

        #print(transitions)
        #print(torch.sum(transitions, dim=1))
        return torch.log(transitions)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Take the emission scores calculated by NERModel, and return a tensor of CRF features,
        which is the sum of transition scores and emission scores.
        :param scores: emission scores calculated by NERModel.
            shape: (batch_size, sentence_length, ntags)
        :return: a tensor containing the CRF features whose shape is
            (batch_size, sentence_len, ntags, ntags). F[b, t, i, j] represents
            emission[t, j] + transition[i, j] for the b'th sentence in this batch.
        """
        batch_size, sentence_len, _ = scores.size()

        # expand the transition matrix batch-wise as well as sentence-wise
        transitions = self.transitions.expand(batch_size, sentence_len, -1, -1)

        # add another dimension for the "from" state, then expand to match
        # the dimensions of the expanded transition matrix above
        emissions = scores.unsqueeze(2).expand_as(transitions)

        # add them up
        return transitions + emissions

    @staticmethod
    def viterbi(features: torch.Tensor) -> torch.Tensor:
        """
        Decode the most probable sequence of tags.
        Note that the delta values are calculated in the log space.
        :param features: the feature matrix from the forward method of CRF.
            shaped (batch_size, sentence_len, ntags, ntags)
        :return: a tensor containing the most probable sequences for the batch.
            shaped (batch_size, sentence_len)
        """
        batch_size, sentence_len, ntags, _ = features.size()

        # initialize the deltas
        delta_t = features[:, 0, START_ID, :]
        deltas = [delta_t]

        # use dynamic programming to iteratively calculate the delta values
        for t in range(1, sentence_len):
            f_t = features[:, t]
            delta_t, _ = torch.max(f_t + delta_t.unsqueeze(2).expand_as(f_t), 1)
            deltas.append(delta_t)

        # now iterate backward to figure out the most probable tags
        sequences = [torch.argmax(deltas[-1], 1, keepdim=True)]
        for t in reversed(range(sentence_len - 1)):
            f_prev = features[:, t + 1].gather(
                2, sequences[-1].unsqueeze(2).expand(batch_size, ntags, 1)).squeeze(2)
            sequences.append(torch.argmax(f_prev + deltas[t], 1, keepdim=True))
        sequences.reverse()
        return torch.cat(sequences, dim=1)