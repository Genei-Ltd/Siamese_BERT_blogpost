def collate(data_list):
    """
        data: list of torchtext.examples
    """
    def pad(seq, max_len):
        # [ [1, 5], [5, 2, 8]  ] ---> [ [1, 5, PAD], [5, 2, 8] ], PAD = 0
        return torch.tensor([item + (max_len - len(item))*[0] for item in seq], dtype=int)

    # get the maximum length of a premise and hypothesis within a batch
    max_premise_len, max_hypothesis_len = torch.tensor( 
    		[ [len(d.premise), len(d.hypothesis)] for d in data_list ] 
    	).max(dim=0).values 

    # pad all premises in the premise batch to the length of the largest premise; do same for hypothesis
    premise = pad([d.premise for d in data_list], max_len=max_premise_len.item())
    hypothesis = pad([d.hypothesis for d in data_list], max_len=max_hypothesis_len.item())

    # map label to number
    label2idx = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    label = torch.tensor([label2idx[d.label] for d in data_list], dtype=int)

    return premise, hypothesis, label