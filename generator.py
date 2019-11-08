import tensorflow as tf
import tensorflow.keras as keras
import ops
class Generator(keras.Model):
    def __init__(self,scope: str="Generator",ngf:int=64,reg:float=0.0005,norm:str="instance",more:bool=True):
        super(Generator, self).__init__(name=scope)
        self.c7s1_32=ops.c7s1_k(scope="c7s1_32",k=ngf,reg=reg,norm=norm)
        self.d64 = ops.dk(scope="d64",k=2*ngf,reg=reg,norm=norm)
        self.d128 = ops.dk(scope="d128",k=4*ngf,reg=reg,norm=norm) 
        if more:
            self.res_output=ops.n_res_blocks(scope="8_res_blocks",n=8,k=4*ngf,reg=reg,norm=norm)
        else:
            self.res_output=ops.n_res_blocks(scope="6_res_blocks",n=6,k=4*ngf,reg=reg,norm=norm)
        self.u64=ops.uk(scope="u64",k=2*ngf,reg=reg,norm=norm)
        self.u32=ops.uk(scope="u32",k=ngf,reg=reg,norm=norm)
        self.outconv = ops.c7s1_k(scope="output",k=3,reg=reg,norm=norm)
    def call(self,x,training=False):
        x = self.c7s1_32(x,training=training,activation='Relu')
        x = self.d64(x,training=training)
        x = self.d128(x,training=training)
        x = self.res_output(x,training=training)
        x = self.u64(x,training=training)
        x = self.u32(x,training=training)
        x = self.outconv(x,training=training,activation='tanh')
        return x
		