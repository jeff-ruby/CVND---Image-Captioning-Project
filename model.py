import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    
class EncoderCNN_Others(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN_Others, self).__init__()

        self.model = models.inception_v3(pretrained=True, aux_logits=True)
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        if self.model.training and self.model.aux_logits:
            features,_ = self.model(images)
        else:
            features = self.model(images)
        #features = features.view(features.size(0), -1)
        #features = self.embed(features)
        features = self.bn(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # define a LSTM model
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,  
                            num_layers=num_layers, batch_first=True)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size

        # embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        # fully connected layer
        self.fc = nn.Linear(self.hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # get the sequence length for further use
        seq_len = captions.size(1)

        # resize for feature tensor from encoder
        features = features.view(features.size(0), 1, self.embed_size)

        # embedding for words of caption
        embeds = self.embed(captions)

        # concatenates features and captions together
        input_seq = torch.cat((features,embeds[:,:-1,:]), dim=1)

        # LSTM model
        # Input size:  batch*seq_len*embed_size
        # hidden size: num_layers*batch*hidden_size
        # output size: batch*seq_len*hidden_size
        self.hidden = self.init_hidden(captions.size(0))
        output, self.hidden = self.lstm(input_seq, self.hidden)
        
        # output score by fully connected layer
        word_scores = self.fc(output.contiguous().view(-1, self.hidden_size))
        
        # resize as: batch*seq_len*vocab_size
        word_scores = word_scores.view(captions.size(0),seq_len,-1)

        return word_scores

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        
    def predict(self, inputs, h, topk=1):
        # output the score vector by LSTM
        output,h = self.lstm(inputs.view(1,1,-1), h)
        score = self.fc(output)

        # calulate the probability by softmax, and pick up the index of max
        p = F.softmax(score.view(1,-1), dim=1)
        p,index = torch.topk(p, topk)

        # return p, index and h,
        # index has the same shape as p(1*topk)
        return p, index, h

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        if states == None:
            h = self.init_hidden(1)

        length = 0
        index = 0
        idx = []

        while length < max_len and index != 1:
            # predict the next word
            _, index, h = self.predict(inputs, h)
            # embedding the index for the next time step
            inputs = self.embed(index.view(-1))

            # add the current index number into the returned list
            length += 1
            index = int(index.cpu().numpy().squeeze())
            idx.append(index)

        return idx
  
    def beam_search_predictions(self, inputs, states=None, beam_index=10, max_len=20):
        " beam search"
        if states == None:
            h = self.init_hidden(1)
        
        # It seem the first output always being <start>
        _, index, h = self.predict(inputs, h)

        # start_word[0][0] = index of the starting word 
        # start_word[0][1] = probability of the word predicted 
        start = [index.view(-1)]
        start_word = [[start, 1.0]]
        hidden_state = {}

        while len(start_word[0][0]) < max_len:
            temp = []
            for s in start_word:
                # embedding
                inputs = self.embed(s[0][-1])
                # get the saved hidden state
                if len(hidden_state) == 0:
                    hidden = h
                else:
                    hidden = hidden_state[s[0][-2]]
                # predict the words
                preds,index,h = self.predict(inputs, hidden, topk=beam_index)
                # save the current hidden state
                hidden_state[s[0][-1]] = h

                # Getting the top <beam_index>(n) predictions 
                word_preds = index.view(-1)

                # change the tensor to numpy
                #word_preds = word_preds.cpu().numpy()
                preds = preds.cpu().detach().numpy()

                # creating a new list so as to put them via the model again 
                for i,w in enumerate(word_preds): 
                    next_cap, prob = s[0][:], s[1] 
                    next_cap.append((w)) 
                    prob *= preds[0][i] 
                    temp.append([next_cap, prob]) 
            
            start_word = temp
            # Sorting according to the probabilities 
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]
            
            '''
            print('Iter')
            for s in start_word:
                print('    P: ', s[1], end=' ')
                print('    Output: ', [int(idx.cpu().numpy()) for idx in s[0]])
            '''
            
        start_word = start_word[-1][0]
        start_word = [int(idx.cpu().numpy()) for idx in start_word]
                
        return start_word