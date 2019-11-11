import tensorflow as tf
import data
import numpy as np
import math
import cv2
import discriminator
import generator
from absl.flags import FLAGS
from absl import app, flags





flags.DEFINE_string('logdir', './log/nofog', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('use_lsgan', True, 'use lsgan (mean squared error) or cross entropy loss, default: True')
flags.DEFINE_integer('batch_size', 1, 'The batch_size for training.')
flags.DEFINE_string('norm', 'instance','[instance, batch] use instance norm or batch norm, default: instance')
flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
flags.DEFINE_integer('epoch', 100,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

flags.DEFINE_float('weight_decay', 0, "The weight decay for ENet convolution layers.")


#loss function

#source is x,target is y	
@tf.function
def train_step(G,F,D_Y,D_X,source,target,generator_g_optimizer,generator_f_optimizer,discriminator_x_optimizer,discriminator_y_optimizer,train_loss,lambda1,lambda2):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = G(source,training=True)
        cycled_x = F(fake_y,training=True)

        fake_x = F(target,training=True)
        cycled_y = G(fake_x,training=True)

        same_x = F(source,training=True)
        same_y = G(target,training=True)
		
        disc_real_x = D_X(source,training=True)
        disc_real_y = D_Y(target,training=True)

        disc_fake_x = D_X(fake_x,training=True)
        disc_fake_y = D_Y(fake_y,training=True)
        #generator loss using lsgan
        gen_g_loss = tf.reduce_mean(tf.math.squared_difference(disc_fake_y, 0.9))
        gen_f_loss = tf.reduce_mean(tf.math.squared_difference(disc_fake_x, 0.9))
        #cycle loss		
        total_cycle_loss = tf.reduce_mean(tf.abs(source-cycled_x))*lambda1 +  tf.reduce_mean(tf.abs(target-cycled_y))*lambda2
        total_gen_g_loss = gen_g_loss + total_cycle_loss + tf.reduce_mean(tf.abs(target - same_y))*0.5*lambda2
        total_gen_f_loss = gen_f_loss + total_cycle_loss + tf.reduce_mean(tf.abs(source - same_x))*0.5*lambda1
        #diss loss
        disc_x_loss = 0.5*tf.reduce_mean(tf.math.squared_difference(disc_real_x,0.9)+tf.math.square(disc_fake_x))
        disc_y_loss = 0.5*tf.reduce_mean(tf.math.squared_difference(disc_real_y,0.9)+tf.math.square(disc_fake_y))		
        #reg loss
        #loss_reg = tf.reduce_sum(G.losses)+tf.reduce_sum(F.losses)+tf.reduce_sum(D_Y.losses)+tf.reduce_sum(D_X.losses)


    train_loss[0](total_gen_g_loss)
    train_loss[1](total_gen_f_loss)
    train_loss[2](disc_x_loss)
    train_loss[3](disc_y_loss)
    generator_g_gradients = tape.gradient(total_gen_g_loss,G.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,F.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss,D_X.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,D_Y.trainable_variables)

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, G.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, F.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,D_X.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,D_Y.trainable_variables))
    

    return fake_y, cycled_x,fake_x,cycled_y,same_x,same_y, total_gen_g_loss,total_gen_f_loss,disc_x_loss,disc_y_loss,generator_g_optimizer.iterations

@tf.function
def lr_sch(epoch):
    if epoch+1<=50:
        lr = tf.constant(2e-4)
    else:
        lr = tf.constant(2e-4-2e-8)*(1-(epoch+1-50)/50)+2e-8
    return lr




def main(_argv):
    weight_decay = FLAGS.weight_decay
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epoch
    log_dir = FLAGS.logdir	
    use_lsgan = FLAGS.use_lsgan 
    norm = FLAGS.norm
    lambda1 = FLAGS.lambda1
    lambda2 = FLAGS.lambda2    
    beta1 = FLAGS.beta1
    ngf = FLAGS.ngf
    G = generator.Generator('G',ngf,weight_decay,norm=norm,more=True)   
    F = generator.Generator('F',ngf,weight_decay,norm=norm,more=True)   
    D_Y = discriminator.Discriminator('D_Y',reg=weight_decay,norm=norm)
    D_X = discriminator.Discriminator('D_X',reg=weight_decay,norm=norm)	
    forbuild=np.random.rand(1,768,768,3).astype(np.float32)
    built=G(forbuild)
    built=F(forbuild)
    built=D_Y(forbuild)
    built=D_X(forbuild)  
	
    source_data=data.source_data(batch_size)
    target_data=data.target_data(batch_size)

    train_loss_G = tf.keras.metrics.Mean('train_loss_G', dtype=tf.float32)
    train_loss_F = tf.keras.metrics.Mean('train_loss_F', dtype=tf.float32)
    train_loss_DX = tf.keras.metrics.Mean('train_loss_DX', dtype=tf.float32)	
    train_loss_DY = tf.keras.metrics.Mean('train_loss_DY', dtype=tf.float32)		
    train_loss = [train_loss_G,train_loss_F,train_loss_DX,train_loss_DY]
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    train_summary_writer = tf.summary.create_file_writer(log_dir)

    ckpt = tf.train.Checkpoint(G=G,F=F,D_X=D_X,D_Y=D_Y,generator_g_optimizer=generator_g_optimizer,generator_f_optimizer=generator_f_optimizer,discriminator_x_optimizer=discriminator_x_optimizer,discriminator_y_optimizer=discriminator_y_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=10)
    start =0
    lr = lr_sch(start)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()	
        start=int(ckpt_manager.latest_checkpoint.split('-')[-1])
        lr = lr_sch(start)
        print ('Latest checkpoint restored!!')
    for ep in range(start,epoch,1):
        print('Epoch:'+str(ep+1))	
        for step,  source in enumerate(source_data):
            target = next(iter(target_data))		
            fake_y, cycled_x,fake_x,cycled_y,same_x,same_y, total_gen_g_loss,total_gen_f_loss,disc_x_loss,disc_y_loss,steps = train_step(G,F,D_Y,D_X,source,target,generator_g_optimizer,generator_f_optimizer,discriminator_x_optimizer,discriminator_y_optimizer,train_loss,lambda1,lambda2)
            print('Step: '+str(steps.numpy())+' , G loss: '+str(total_gen_g_loss.numpy())+' , F loss: '+str(total_gen_f_loss.numpy())+' , D_X loss: '+str(disc_x_loss.numpy())+' , D_Y loss: '+str(disc_y_loss.numpy()))
            if (steps.numpy()-1)%10==0:
                source=tf.image.convert_image_dtype((source+1)/2,dtype = tf.uint8)
                target=tf.image.convert_image_dtype((target+1)/2,dtype = tf.uint8)			
                fake_y=tf.image.convert_image_dtype((fake_y+1)/2,dtype = tf.uint8)
                cycled_x=tf.image.convert_image_dtype((cycled_x+1)/2,dtype = tf.uint8)
                same_x=tf.image.convert_image_dtype((same_x+1)/2,dtype = tf.uint8)
                fake_x=tf.image.convert_image_dtype((fake_x+1)/2,dtype = tf.uint8)
                cycled_y=tf.image.convert_image_dtype((cycled_y+1)/2,dtype = tf.uint8)
                same_y=tf.image.convert_image_dtype((same_y+1)/2,dtype = tf.uint8)
                with train_summary_writer.as_default():
                    tf.summary.scalar('Learning_rate', lr, step=steps)
                    tf.summary.scalar('Loss/G_loss', train_loss[0].result(), step=steps)
                    tf.summary.scalar('Loss/F_loss', train_loss[1].result(), step=steps)
                    tf.summary.scalar('Loss/DX_loss', train_loss[2].result(), step=steps)
                    tf.summary.scalar('Loss/DY_loss', train_loss[3].result(), step=steps)
                    tf.summary.image('image_source/source',source,max_outputs=1,step=steps)
                    tf.summary.image('image_source/fake_target',fake_y,max_outputs=1, step=steps)
                    tf.summary.image('image_source/cycle_source',cycled_x,max_outputs=1,step=steps)
                    tf.summary.image('image_source/same_source',same_x,max_outputs=1, step=steps)
                    tf.summary.image('image_target/target',target,max_outputs=1,step=steps)
                    tf.summary.image('image_target/fake_source',fake_x,max_outputs=1, step=steps)
                    tf.summary.image('image_target/cycle_target',cycled_y,max_outputs=1,step=steps)
                    tf.summary.image('image_target/same_target',same_y,max_outputs=1, step=steps)
                train_loss[0].reset_states()
                train_loss[1].reset_states()
                train_loss[2].reset_states()
                train_loss[3].reset_states()
        lr = lr_sch(ep)
        generator_g_optimizer.learning_rate = lr
        generator_f_optimizer.learning_rate = lr
        discriminator_x_optimizer.learning_rate = lr
        discriminator_y_optimizer.learning_rate = lr
        ckpt_save_path = ckpt_manager.save()
    print("Traing is over!")		
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass	

