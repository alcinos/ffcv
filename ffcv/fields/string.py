from .ndarray import NDArrayField
import numpy as np

class StringField(NDArrayField):
    def __init__(self, max_len: int, pad_char='\0'):
        self.max_len = max_len
        self.pad_char = pad_char
        super().__init__(np.dtype('uint8'), (max_len,))
    
    def encode(self, destination, field, malloc):
        padded_field = (field + self.pad_char * self.max_len)[:self.max_len]
        field = np.frombuffer(padded_field.encode('ascii'), dtype='uint8')
        return super().encode(destination, field, malloc)
    
