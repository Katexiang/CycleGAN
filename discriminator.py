import tensorflow as tf
import tensorflow.keras as keras
import ops
class Discriminator(keras.Model):
    def __init__(self,scope: str="Discriminator",reg:float=0.0005,norm:str="instance"):
        super(Discriminator, self).__init__(name=scope)
        self.ck1 = ops.Ck(scope="C64",k=64,reg=reg,norm=norm)
        self.ck2 = ops.Ck(scope="C128",k=128,reg=reg,norm=norm)
        self.ck3 = ops.Ck(scope="C256",k=256,reg=reg,norm=norm)
        self.ck4 = ops.Ck(scope="C512",k=512,reg=reg,norm=norm)
        self.last_conv = ops.last_conv(scope="output",reg=reg)
    def call(self,x,training=False,use_sigmoid=False,slope=0.2):
        x=self.ck1(x,training=training,slope=slope)
        x=self.ck2(x,training=training,slope=slope)
        x=self.ck3(x,training=training,slope=slope)
        x=self.ck4(x,training=training,slope=slope)
        x=self.last_conv(x,use_sigmoid=use_sigmoid)
        return x