import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, heads_tails,ManyDatasetsInOne
from metrics import calc_mrr, freeze_model, unfreeze_model
from models import RGCN
from args import args
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random

#Add
from GAN import *
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, average_precision_score, ndcg_score, dcg_score

def train(Tensor,LongTensor, adversarial_loss, auxiliary_loss, distmult_loss, gen_loss, gen_loss2,  dis_loss, epoch, train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    # Extract subgraph, then agg_embed from RGCN
    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities,
                                                   num_relations, negative_sample)
    a = train_data.edge_type
    b = train_data.edge_index
    c = train_data.labels

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type,
                             train_data.edge_norm)

    pos_head, pos_rel, pos_tail = model.pos_embedding(entity_embedding, train_data.relabeled_edges.long())
    custom_data = ManyDatasetsInOne(pos_head, pos_rel, pos_tail)

    inverse_relation = 1/pos_rel

    train_ldr = torch.utils.data.DataLoader(custom_data,
                                        batch_size=1000, shuffle=False, drop_last=True)

    args.dis_batch_loss = 0
    args.gen_batch_loss = 0
    args.gen_batch_loss2 = 0
    freeze_model(model)
    for (batch_idx, real_data) in enumerate(train_ldr):
        # Adversarial ground truths
        valid = Variable(Tensor(real_data[0].size(0), 1).fill_(1.0), requires_grad=False)


        pos_head_batch = real_data[0].cuda().detach()
        pos_rel_batch = real_data[1].cuda().detach()
        inverse_relation_batch = 1/pos_rel_batch
        pos_tail_batch = real_data[2].cuda().detach()


        # Train Generator
        if epoch < args.freeze_epoch :
            unfreeze_model(args.gen)
            # unfreeze_model(args.gen2)
            freeze_model(args.dis)
            for dis_idx in range(args.epoch_g):
                args.generator_optimizer.zero_grad()

                fake_tail1,filt_l2_regularization_tail = args.gen(pos_head_batch)
                fake_head1,filt_l2_regularization_head = args.gen(pos_tail_batch)

                l2_regularization = (filt_l2_regularization_tail  + filt_l2_regularization_head)/2

                # Loss measures generator's ability to fool the discriminator
                validity1, _ =args.dis(fake_tail1)
                g_loss_tail = adversarial_loss(validity1, valid)
                validity2, _ =args.dis(fake_head1)
                g_loss_head = adversarial_loss(validity2, valid)

                g_loss = (g_loss_tail + g_loss_head)/2 + args.reg*l2_regularization

                args.gen_batch_loss += g_loss.detach().cpu().numpy().item()

                g_loss.backward()
                args.generator_optimizer.step()

        # Train Generator2
        if epoch < args.freeze_epoch:
            #unfreeze_model(args.gen)
            unfreeze_model(args.gen2)
            freeze_model(args.dis)
            for dis_idx in range(args.epoch_g):
                args.generator_optimizer2.zero_grad()

                # args.gen.train()
                fake_tail2,filt_l2_regularization_tail = args.gen2(pos_head_batch,pos_rel_batch)
                fake_head2,filt_l2_regularization_head  = args.gen2(pos_tail_batch,inverse_relation_batch)

                l2_regularization = (filt_l2_regularization_tail + filt_l2_regularization_head)/2

                # Loss measures generator's ability to fool the discriminator
                validity1, _ =args.dis(fake_tail2)
                g_loss_tail = adversarial_loss(validity1, valid)
                validity2, _ =args.dis(fake_head2)
                g_loss_head = adversarial_loss(validity2, valid)

                g_loss2 = (g_loss_tail + g_loss_head)/2 + args.reg*l2_regularization

                args.gen_batch_loss2 += g_loss2.detach().cpu().numpy().item()

                g_loss2.backward()
                args.generator_optimizer2.step()


        # Train Discriminator

        #Merge fake1 & fake2
        combinedFake_tail = torch.cat((fake_tail1,fake_tail2),0)
        combinedFake_head = torch.cat((fake_head1, fake_head2), 0)

        #Merge label of fake1 & fake2
        fake1 = Variable(Tensor(real_data[0].size(0), 1).fill_(1.0), requires_grad=False)
        fake2 = Variable(Tensor(real_data[0].size(0), 1).fill_(0.0), requires_grad=False)
        fake_classlabels = torch.cat((fake1, fake2), 0)
        fakelabels = Variable(Tensor(real_data[0].size(0)*2, 1).fill_(0.0), requires_grad=False)
        # fakelabels2 = Variable(Tensor(real_data[0].size(0), 1).fill_(0.0), requires_grad=False)

        if epoch < args.freeze_epoch :
            freeze_model(args.gen)
            freeze_model(args.gen2)
            unfreeze_model(args.dis)
            for dis_idx in range(args.epoch_d):
                args.discriminator_optimizer.zero_grad()

                # Loss for real head/tail
                ##OurModel : Predict real as 1 and real as 3
                real_pred, real_aux = args.dis(pos_tail_batch)
                d_real_loss = adversarial_loss(real_pred, valid)

                real_pred1, real_aux1 = args.dis(pos_head_batch)
                d_real_loss1 = adversarial_loss(real_pred1, valid)

                # Loss for fake tail
                fake_pred, fake_aux = args.dis(combinedFake_tail.detach())
                d_fake_loss = (adversarial_loss(fake_pred, fakelabels) + auxiliary_loss(fake_aux, fake_classlabels)) / 2

                # Loss for fake head
                fake_pred2, fake_aux2 = args.dis(combinedFake_head.detach())
                d_fake_loss_1 = (adversarial_loss(fake_pred2, fakelabels) + auxiliary_loss(fake_aux2, fake_classlabels)) / 2

                d_loss = (d_real_loss + d_real_loss1 + d_fake_loss + d_fake_loss_1)/4
                args.dis_batch_loss += d_loss.detach().cpu().numpy().item()


                d_loss.backward()
                args.discriminator_optimizer.step()


    freeze_model(args.gen)
    freeze_model(args.gen2)
    freeze_model(args.dis)
    unfreeze_model(model)
    model.train()
    #Using Generator to produce fake nodes
    fake_head1 = torch.empty(0, args.node_embed_size).cuda()
    fake_tail1 = torch.empty(0, args.node_embed_size).cuda()
    fake_tail2 = torch.empty(0, args.node_embed_size).cuda()
    fake_head2 = torch.empty(0, args.node_embed_size).cuda()
    for i in range(args.gen_fake_ratio):
        fake_o,_ = args.gen(pos_head.cuda())
        fake_s,_ = args.gen(pos_tail.cuda())
        fake_o2,_ = args.gen2(pos_head.cuda(), pos_rel.cuda())
        fake_s2,_ = args.gen2(pos_tail.cuda(), inverse_relation.cuda())
        fake_tail1 = torch.cat((fake_tail1, fake_o), dim=0)
        fake_tail2 = torch.cat((fake_tail2, fake_o2), dim=0)
        fake_head1 = torch.cat((fake_head1, fake_s), dim=0)
        fake_head2 = torch.cat((fake_head2, fake_s2), dim=0)


    #Feed Fake triplet + random negative triplet to DistMult
    RGCN_loss = model.score_loss(entity_embedding,fake_tail1,fake_tail2, fake_head1,fake_head2 , train_data.samples, train_data.labels, train_data.relabeled_edges) + reg_ratio * model.reg_loss(entity_embedding)
    args.optimizer.zero_grad()
    RGCN_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
    args.optimizer.step()
    
    genloss = args.gen_batch_loss/(len(train_ldr)*args.epoch_g)
    gen_loss.append(genloss)
    tqdm.write("Total epoch: {}, Gen Loss: {}.".
               format(epoch, genloss))

    genloss2 = args.gen_batch_loss2/(len(train_ldr)*args.epoch_g)
    gen_loss2.append(genloss2)

    tqdm.write("Total epoch: {}, Gen Loss 2: {}.".
               format(epoch, genloss2))

    disloss = args.dis_batch_loss/(len(train_ldr)*args.epoch_d)
    dis_loss.append(disloss)


    tqdm.write("Total epoch: {}, Discriminator loss: {}.".
               format(epoch, disloss))

    distmult = RGCN_loss.detach().cpu().numpy().item()
    distmult_loss.append(distmult)
    tqdm.write("Total epoch: {}, DisMult Loss: {}.".
               format(epoch, RGCN_loss))


    return RGCN_loss

def valid(valid_data, model, test_graph, n_ent, filt_heads,filt_tails):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(valid_data, n_ent, filt_heads, filt_tails, entity_embedding, model.relation_embedding )




    return mrr

def test(test_data, model, test_graph, n_ent, filt_heads,filt_tails):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(test_data, n_ent, filt_heads, filt_tails, entity_embedding, model.relation_embedding)

    return mrr


def main(args):

    #Set Pytorch Seed
    random_seed = 123
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)


    best_mrr = 0

    entity2id, relation2id, train_triplets, valid_triplets, test_triplets_o = load_data('./data/NELL-995')
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets_o)))
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets_o)

    #Define RGCN, GAN
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, dropout=args.dropout)


    #Added
    all_data = all_triplets.numpy()
    #filt_heads, filt_tails = heads_tails(len(entity2id), all_data)
    filt_heads, filt_tails = heads_tails(len(entity2id), all_data)
    valid_data = valid_triplets.numpy()
    valid_src, valid_rel, valid_dst = valid_data.transpose().tolist()
    valid_data = valid_src, valid_rel, valid_dst
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data = test_triplets.numpy()
    test_src, test_rel, test_dst = test_data.transpose().tolist()
    test_data = test_src, test_rel, test_dst
    test_data = [torch.LongTensor(vec) for vec in test_data]

    gen,gen2, dis,filt,filt2= WayGAN2(args=args, model_path=None).getVariables2()
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    args.gen = gen.cuda()
    args.gen2 = gen2.cuda()
    args.filt = filt.cuda()
    args.filt2 = filt2.cuda()
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    args.dis = dis.cuda()

    args.dis.train()  # set mode
    args.gen.train()
    args.gen2.train()
    args.discriminator_optimizer = torch.optim.Adam(args.dis.parameters(), lr=args.dis_lr, betas=(args.b1, args.b2))
    args.generator_optimizer = torch.optim.Adam(args.gen.parameters(), lr=args.gen_lr, betas=(args.b1, args.b2))
    args.generator_optimizer2 = torch.optim.Adam(args.gen2.parameters(), lr=args.gen_lr, betas=(args.b1, args.b2))
    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.rgcn_lr)

    #Load pretrained RGCN
    checkpoint = torch.load(''./Pretrained/pretrained_best_mrr_model_NELL.pth'')
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    print('pretrain epoch', epoch)
    loss = checkpoint['loss']
    print('pretrain loss', loss)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    valid_mrr_plot = []
    dis_loss = []
    gen_loss = []
    gen_loss2 = []
    distmult_loss = []

    print(model)

    if use_cuda:
        model.cuda()

    for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):
        RGCN_loss = train(Tensor,LongTensor, adversarial_loss, auxiliary_loss, distmult_loss, gen_loss, gen_loss2, dis_loss, epoch, train_triplets, model, use_cuda, batch_size=args.graph_batch_size,  split_size=args.graph_split_size,
            negative_sample=args.negative_sample, reg_ratio = args.regularization, num_entities=len(entity2id), num_relations=len(relation2id))

        if epoch % args.evaluate_every == 0:

            if use_cuda:
                model.cpu()

            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            model.eval()
            with torch.no_grad():
                valid_mrr = valid(valid_data, model, test_graph, len(entity2id), filt_heads, filt_tails)
                # print('valid_mrr', valid_mrr)
            valid_mrr = valid_mrr.detach().cpu().numpy().item()
            valid_mrr_plot.append(valid_mrr)



            
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': args.optimizer.state_dict(),
                            'loss': RGCN_loss},
                            'best_mrr_model.pth')
                
            if use_cuda:
                model.cuda()
    
    if use_cuda:
        model.cpu()

    random_seed = 123
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()

    checkpoint = torch.load('best_mrr_model.pth')
    model.load_state_dict(checkpoint['state_dict'])
    test(test_data, model, test_graph, len(entity2id), filt_heads,filt_tails)



if __name__ == '__main__':

    print(args)
    main(args)
