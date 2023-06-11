import numpy as np
import tensorflow as tf
import random
import sys, os
from core.helpers import Parser, Dataset
from core.engines import PolicyNet, SelfAttentionNet
import time
import logging


def adjusting_weight(lenth, orin_weight, actions, beta):
    ori_ = list(orin_weight)
    to_lower_ind, new_lowered, to_keep_ind, to_higher_ind, new_highered = [], [], [], [], []
    changed_dict = {}
    sum_ori_all = float(sum(ori_))
    sum_ori_keep = .0

    for i in range(len(ori_)):
        item = ori_[i]
        if i < lenth:
            act = actions[i]
            if act == 0:
                to_lower_ind.append(i)
                new_lowered.append(item * (1 - beta))
            elif act == 1:
                to_higher_ind.append(i)
                new_highered.append(item * (1 + beta))
        else:
            to_keep_ind.append(i)
            sum_ori_keep += item
            changed_dict[i] = item

    sum_var = sum_ori_all - sum_ori_keep
    sum_to_chage = float(sum(new_lowered)) + float(sum(new_highered))
    new_lowered = [(item * sum_var / sum_to_chage) for item in new_lowered]
    new_highered = [(item * sum_var / sum_to_chage) for item in new_highered]
    for val, i in zip(new_lowered, to_lower_ind):
        changed_dict[i] = val
    for val, i in zip(new_highered, to_higher_ind):
        changed_dict[i] = val
    new_weight = [changed_dict[i] for i in range(len(ori_))]
    return np.array(new_weight)


def do_steps(actor, vec, lenth, A_, M_, H_, epsilon=0., tag=None, Random=True):
    actions = []
    states = []
    ori_real_sumed_attentioned_vec = H_.sum(axis=0)

    for pos in range(lenth):
        states.append([M_[pos], H_[pos], np.array([vec[0][pos]]),  #
                       ori_real_sumed_attentioned_vec])
        predicted = actor.predict_target(M_[pos], H_[pos], [vec[0][pos]],
                                         ori_real_sumed_attentioned_vec)

        if Random:
            if random.random() < epsilon:
                action = (0 if random.random() < predicted[0] else 1)
            else:
                action = (1 if random.random() < predicted[0] else 0)

        else:
            action = np.argmax(predicted)
        actions.append(action)

    len_changing_vars = len([a for a in actions if a == 1])

    new_weight = adjusting_weight(lenth, A_, actions, args.beta)
    flatten_H = np.transpose(H_, (1, 0, 2))
    new_weight = np.expand_dims(np.expand_dims(new_weight, axis=0), axis=2)
    attented_vec = flatten_H * new_weight

    return actions, states, attented_vec, len_changing_vars


def train(actor, critic, train_data, batchsize, samplecnt=5, Attention_trainable=True, RL_trainable=True):
    random.shuffle(train_data)
    for b in range(len(train_data) // batchsize):
        datas = train_data[b * batchsize: (b + 1) * batchsize]
        totloss = 0.
        critic.assign_active_network()
        actor.assign_active_network()

        for j in range(batchsize):
            data = datas[j]
            inputs, solution, lenth = data['words'], data['solution'], data['lenth']
            if RL_trainable:
                actionlist, statelist, losslist = [], [], []
                A_, M_, H_ = critic.get_A_M_H([lenth], [inputs])
                sentence_wv = critic.wordvector_find([inputs])

                aveloss = 0.
                for i in range(samplecnt):
                    actions, states, attented_vec, len_changing_vars = do_steps(actor, sentence_wv,
                                                                                lenth, A_, M_, H_,
                                                                                args.epsilon, tag=None,
                                                                                Random=True)
                    actionlist.append(actions)
                    statelist.append(states)
                    out, loss = critic.getloss_with_rl_att([lenth], [inputs], attented_vec, [solution])
                    loss += (float(len_changing_vars) / lenth) ** 2 * 0.15
                    aveloss += loss
                    losslist.append(loss)

                aveloss /= samplecnt
                totloss += aveloss
                grad = None
                if Attention_trainable:
                    out, loss, _ = critic.train_with_rl_att([lenth], [inputs], attented_vec, [solution])
                for i in range(samplecnt):
                    for pos in range(len(actionlist[i])):
                        rr = [0., 0.]
                        rr[actionlist[i][pos]] = (losslist[i] - aveloss) * args.alpha
                        g = actor.get_gradient(statelist[i][pos][0], statelist[i][pos][1], statelist[i][pos][2],
                                               statelist[i][pos][3], rr)
                        if grad == None:
                            grad = g
                        else:
                            grad[0] += g[0]
                            grad[1] += g[1]
                            grad[2] += g[2]
                            grad[3] += g[3]
                            grad[4] += g[4]
                actor.train(grad)
            else:
                out, loss, _ = critic.train_without_rl_att([lenth], [inputs], [solution])
                totloss += loss

        if RL_trainable:
            actor.update_target_network()
            if Attention_trainable:
                critic.update_target_network()
        else:
            critic.assign_target_network()

        if (b + 1) % 100 == 0:
            acc_dev = test(actor, critic, dev_data, noRL=not RL_trainable)
            if RL_trainable and Attention_trainable:
                print("batch ", b, "total loss ", totloss, "| dev: ", acc_dev)
                logger.info(
                    "batch " + str(b) + "total loss " + str(totloss) + "| dev: " + str(
                        acc_dev))


def test(actor, critic, val_data, noRL=False):
    acc = 0
    for i in range(len(val_data)):
        data = val_data[i]
        inputs, solution, lenth = data['words'], data['solution'], data['lenth']

        if noRL:
            out = critic.predict_target_without_rl_att([lenth], [inputs])
        else:
            A_, M_, H_ = critic.get_A_M_H([lenth], [inputs])
            sentence_wv = critic.wordvector_find([inputs])
            actions, states, attented_vec, len_changing_vars = do_steps(actor,
                                                                        sentence_wv,
                                                                        lenth,
                                                                        A_, M_,
                                                                        H_,
                                                                        args.epsilon,
                                                                        Random=False)
            out = critic.predict_target_with_rl_att([lenth], [inputs], attented_vec)
        if np.argmax(out) == np.argmax(solution):
            acc += 1
    return float(acc) / len(val_data)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    argv = sys.argv[1:]
    args, _ = Parser().getParser().parse_known_args(argv)
    random.seed()

    logging.basicConfig(filename=args.log, level=logging.INFO,
                        format='%(message)s')
    logger = logging.getLogger(__name__)

    dataset = Dataset(args.dataset, logger)
    train_data, dev_data = dataset.getdata(args.maxlenth)
    word_vector = dataset.get_wordvector(args.word_vector)

    ###
    train_text_num = 500
    dev_text_num = 20
    if args.smalldata == 1:
        train_data = train_data[:train_text_num]
        dev_data = dev_data[:dev_text_num]
    print("train_data ", len(train_data))
    print("dev_data", len(dev_data))
    ###

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Actor-Critic model
        critic = SelfAttentionNet(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.maxlenth,
                        args.dropout, word_vector, logger)
        actor = PolicyNet(sess, args.maxlenth, args.dim, args.optimizer, args.lr, args.tau, logger)

        sess.run(tf.initialize_local_variables())
        sess.run(tf.global_variables_initializer())
        ###############################

        print('*' * 50)
        logger.info('*' * 50)

        for item in tf.trainable_variables():
            print(item.name, item.get_shape())
            logger.info(item.name + str(item.get_shape()))

        num_trainable_vars_len, num_other_variables_len = critic.get_num_trainable_vars()

        print('num_trainable_vars:', num_trainable_vars_len)
        print('num_other_variables:', num_other_variables_len)
        logger.info('num_trainable_vars:' + str(num_trainable_vars_len))
        logger.info('num_other_variables:' + str(num_other_variables_len))

        saver = tf.train.Saver()
        print('*' * 50)
        logger.info('*' * 50)

        ###############################
        # Attention pretrain
        if args.RL_pretrain != '':
            pass
        elif args.Attention_pretrain == '':
            print("start Attention pretraining")
            logger.info("start Attention pretraining")
            epos_start_time = time.time()
            for i in range(0, 2):
                start_time = time.time()
                train(actor, critic, train_data, args.batchsize, args.sample_cnt, RL_trainable=False)
                critic.assign_target_network()
                acc_dev = test(actor, critic, dev_data, True)
                print("Attention_only ", i, "| dev: ", acc_dev)
                logger.info("Attention_only " + str(i) + "| dev: " + str(acc_dev))
                saver.save(sess, "checkpoints/" + args.name + "_Attentionpre", global_step=i)
                print("Attention pretrain time spend for round %i: %.4f min" % (i, (time.time() - start_time) / 60))
                logger.info(
                    "Attention pretrain time spend for round %i: %.4f min" % (i, (time.time() - start_time) / 60))
                print()
                logger.info('')
            print(
                "Attention pretrain done, time spend for all rounds: %.4f min" % ((time.time() - epos_start_time) / 60))
            logger.info(
                "Attention pretrain done, time spend for all rounds: %.4f min" % ((time.time() - epos_start_time) / 60))
        else:
            print("Load Attention from ", args.Attention_pretrain)
            logger.info("Load Attention from " + str(args.Attention_pretrain))
            saver.restore(sess, args.Attention_pretrain)
        print('*' * 50)
        logger.info('*' * 50)

        ###############################
        # RL pretrain
        if args.RL_pretrain == '':
            print("start RL pretraining")
            logger.info("start RL pretraining")
            epos_start_time = time.time()
            for i in range(0, 3):
                start_time = time.time()
                train(actor, critic, train_data, args.batchsize, args.sample_cnt, Attention_trainable=False)
                acc_dev = test(actor, critic, dev_data)
                print("RL pretrain ", i, "| dev: ", acc_dev)
                logger.info("RL pretrain " + str(i) + "| dev: " + str(acc_dev))
                saver.save(sess, "checkpoints/" + args.name + "_RLpre", global_step=i)
                print("RL pretrain time spend for round %i: %.4f min" % (i, (time.time() - start_time) / 60))
                logger.info("RL pretrain time spend for round %i: %.4f min" % (i, (time.time() - start_time) / 60))
                logger.info('')
                print()
            print("RL pretrain done, time spend for all rounds: %.4f min" % ((time.time() - epos_start_time) / 60))
            logger.info(
                "RL pretrain done, time spend for all rounds: %.4f min" % ((time.time() - epos_start_time) / 60))
        else:
            print("Load RL from", args.RL_pretrain)
            saver.restore(sess, args.RL_pretrain)
        print('*' * 50)
        logger.info('*' * 50)

        ###############################
        # RL critic-Attention jointly training
        epos_start_time = time.time()
        print("start co-training")
        logger.info("start co-training")
        for e in range(args.epoch):
            start_time = time.time()
            train(actor, critic, train_data, args.batchsize, args.sample_cnt)
            acc_dev = test(actor, critic, dev_data)
            print("epoch ", e, "| dev: ", acc_dev)
            logger.info("epoch " + str(e) + "| dev: " + str(acc_dev))
            saver.save(sess, "checkpoints/" + args.name, global_step=e)
            print("co-train time spend for round %i: %.4f min" % (e, (time.time() - start_time) / 60))
            logger.info("co-train time spend for round %i: %.4f min" % (e, (time.time() - start_time) / 60))
            logger.info('')
            print()
        print("co-train done, time spend for all rounds: %.4f min" % ((time.time() - epos_start_time) / 60))
        logger.info("co-train done, time spend for all rounds: %.4f min" % ((time.time() - epos_start_time) / 60))
        print('*' * 50)
        logger.info('*' * 50)
