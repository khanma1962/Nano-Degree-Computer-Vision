import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)  # calling pre-trained model
        for param in resnet.parameters():
            param.requires_grad_(False) # freezing all the parameters
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, 
                 num_layers=1, dropout =.05):
        
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embed_size, hidden_size, 
                            num_layers, batch_first = True, 
                            dropout = dropout)
        self.embedded = nn.Embedding(vocab_size, embed_size)
        self.fc  = nn.Linear(hidden_size, vocab_size)
            
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, features, captions):
        
        #output of encoder, features will be (10,256), features[0] will give batch size
        batch_size = features[0]
        
        # from caption removing the end word
        captions_no_end = captions[: , :-1]
        captions = self.embedded(captions_no_end)
        
        # add one more axis to features
        features = torch.unsqueeze(features, 1)
        
        #concatenate features and captions
        concatenated_input = torch.cat((features, captions), 1)
        
        #output from the lstm will be
        lstm_out, _ = self.lstm(concatenated_input, None)
        
        #final output from the fc
        captions_out = self.fc(lstm_out)
        
        return captions_out
        

    def sample(self, inputs, states=None, max_len=20, stop_idx=1):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        lstm_state = None
        caption = []
        for i in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            
            output = self.fc(lstm_out)
            
            #prediction
            pred_val = torch.argmax(output, dim=2)
            pred_index = pred_val.item()
            caption.append(pred_index)
            
            if pred_index == stop_idx:
                break
                
            #input for the next cycle
            inputs = self.embedded(pred_val)
            
            
        return caption    
    
    
    
    
    