## BASELINE - LSTM 

from keras.models import Model 
from keras.layers import Input, Dense, LSTM, Dropout, Masking 

class Network(Model):
    def __init__(self, window_size, output_size, input_dim, output_dim, depth, units=64, activation='tanh', dropout=0.3, rec_dropout=0.3, **kwargs):
        
        self.window_size = window_size
        self.output_size = output_size
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.depth = depth 
        self.units = units 
        self.activation = activation
        self.dropout = dropout 
        self.rec_dropout = rec_dropout

        input_layer = Input(shape=(window_size, input_dim))
        mask = Masking()(input_layer)
        
        for i in range(depth - 1): 
            lstm = LSTM(units=units, 
                        activation=activation,
                        return_sequences=True, 
                        recurrent_dropout=rec_dropout, 
                        dropout=dropout)
            mask = lstm(mask)
        
        L = LSTM(units=units, 
                 activation=activation, 
                 return_sequences=False,
                 dropout=dropout, 
                 recurrent_dropout=rec_dropout)(mask)
        L = Dropout(0.3)(L)
        y = Dense(output_dim, activation='linear')(L)
        
        inputs = [input_layer]
        outputs = [y]
        
        super(Network, self).__init__(inputs=inputs, outputs=outputs) # same syntax as parent Model 
    
    def sayName(self):
        
        return "{}.n{}{}{}".format('k_lstm',
                                        self.input_dim,
                                        ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                        ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "")
        
    def trainModel(self, tr_X, tr_Y, epochs=5, **kwargs):
        
        self.compile(loss='mse', optimizer='adam')
        self.fit(tr_X, tr_Y, epochs=epochs, validation_split=0.1)

    def predictModel(self, te_X): 
        
        return self.predict(te_X)
    
if __name__ == '__main__':
    pass 