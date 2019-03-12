        # Implement your network here
            # equation or predefiend fuctions --convolution operation
            # we define set of layers according to the max_pooling ksize
            # each set has more than one convolution and max_poolin layers
            # totaly we have 2 sets and 5 blocks
            # init phase
            # [filter_h,filter_w,in_channel,out_channel]
            shape1 = [3, 3, 1, 1]
            w = mf.weight_dict(shape1,'init_weight')
            b = mf.bias_dict([shape1[3]],'init_bias')  # out_channel== n_hidden
            # print(X)
            conv1 = mf.conv2d(X, w,'first_conv_layer') + b    # init_convolution
            # -----------1st set--------{2 blocks}---------------------------------------
            with tf.variable_scope("Set-1"):
                # -------1st block
                with tf.name_scope("Block-1.1"):
                    conv_l1 = mf.conv_layer(conv1, [3, 3, 1, 4],'conv_layer-1.1.1')
                    batch_norm1 = mf.batch_n(conv_l1,'batch_norm-1.1.1')  # batch normalization
                    conv_l2 = mf.conv_layer(batch_norm1, [3, 3, 4, 4],'conv_layer-1.1.2')
                    batch_norm2 = mf.batch_n(conv_l2,'batch_norm-1.1.2')           # batch normalization

                    mpool_1 = mf.max_pool(batch_norm2, 1, 1,'maxpool-1.1')   # stride =1 , k=1
    #             # ------2nd block
    #             with tf.name_scope("Block-1.2"):
    #                 conv_l3 = mf.conv_layer(mpool_1, [3, 3, 4, 8],'conv_layer-1.2.1')
    #                 conv_l3 = tf.nn.dropout(conv_l3,keep_prob,name='drop_l-1.2.1') # dropout
    #                 batch_norm3 = mf.batch_n(conv_l3,'batch_norm-1.2.1')  # batch normalization
    #                 conv_l4 = mf.conv_layer(batch_norm3, [3, 3, 8, 8],'conv_layer-1.2.2')
    #                 conv_l4 = tf.nn.dropout(conv_l4,keep_prob,name='drop_l-1.2.2')  # dropout l4
    #                 batch_norm4 = mf.batch_n(conv_l4,'batch_norm-1.2.2')  # batch normalization
    #
    #                 mpool_2 = mf.max_pool(batch_norm4, 1, 1,'max_pool-1.2')   # stride =1 , k=1
    # # --------2nd set------{3 blocks}--------------------------------------------
    #         with tf.name_scope("Set-2"):
    #             # -------3d block
    #             with tf.name_scope("Block-2.1"):
    #                 conv_l5 = mf.conv_layer(mpool_2, [3, 3, 8, 16],'conv_layer-2.1.1')
    #                 batch_norm5 = mf.batch_n(conv_l5,'batch_norm-2.1.1')      # normalization
    #                 conv_l6 = mf.conv_layer(batch_norm5, [3, 3, 16, 16],'conv_layer-2.1.2')
    #                 conv_l6 = tf.nn.dropout(conv_l6,keep_prob,name='drop_l-2.1.2') # dropout
    #                 batch_norm6 = mf.batch_n(conv_l6,'batch_norm-2.1.2')      # normalization batch
    #
    #                 mpool_3 = mf.max_pool(batch_norm6, 1, 1,'max_pool-2.1')   # stride =1 , k=1
    #             # --------4th block
    #             with tf.name_scope("Block-2.2"):
    #                 conv_l7 = mf.conv_layer(mpool_3, [3, 3, 16, 32],'conv_layer-2.2.1')
    #                 conv_l7 = tf.nn.dropout(conv_l7,keep_prob,name='drop_l-2.2.1')
    #                 batch_norm7 = mf.batch_n(conv_l7,'batch_norm-2.2.1')
    #                 conv_l8 = mf.conv_layer(batch_norm7, [3, 3, 32, 32],'conv_layer-2.2.2')
    #                 conv_l8 = tf.nn.dropout(conv_l8,keep_prob,name='drop_l-2.2.2')# dropout
    #                 batch_norm8 = mf.batch_n(conv_l8,'batch_norm-2.2.2')
    #
    #                 mpool_4 = mf.max_pool(batch_norm8, 2, 2,'max_pool-2.2')   # stride=2, k=2
    #             # --------5th blocks
    #             with tf.name_scope("Block-2.3"):
    #                 conv_l9 = mf.conv_layer(mpool_4, [3, 3, 32, 64],'conv_layer-2.3.1')
    #                 conv_l9 = tf.nn.dropout(conv_l9,keep_prob,name='drop_l-2.3.1')
    #                 batch_norm9 = mf.batch_n(conv_l9,'batch_norm-2.3.1')
    #                 conv_l10 = mf.conv_layer(batch_norm9, [3, 3, 64, 64],'conv_layer-2.3.2')
    #                 batch_norm10 = mf.batch_n(conv_l10,'batch_norm-2.3.2')
    #
    #                 mpool_5 = mf.max_pool(batch_norm10, 2, 2,'max_pool-2.3')      # stride=2, k=2
                    # print("SHAPE../n")
                    # print(mpool_5.shape)
                    # print(mpool_5.shape[1].value)

    # ------------add dense layers {4 layers}-------------------------------------
    # flatten out & last layer must not have relu activation fuction
            with tf.name_scope("Dense-Layer"):
                flatt_out=mf.flatten_l(mpool_1,'flatten_l')        # flatten out tensor from 4D to 2D
                l=mf.fully_con(flatt_out,256,'fc1')          # 1st dense-relu layer
                # l=mf.fully_con(l,512,'fc2')
                # l=mf.dense_layer(l,self.n_classes,'last_dense_layer')                  # 2nd
