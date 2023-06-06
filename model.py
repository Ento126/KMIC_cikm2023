import numpy as np
import torch
from torch import nn
from aggregator import Aggregator
from context import Context
from transform import Transform
from transE.transE_embedding_interface import out_embdding_entity, out_embdding_relation

class KMIC(torch.nn.Module):
    def __init__(self, n_user, n_item, list_users, list_items, n_entity, n_relation, adj_entity, adj_relation, user_click_item_pos, args, device, transE_model):
        super(KMIC, self).__init__()
        self.device = device

        self.num_user = n_user 
        self.num_ent = n_entity 
        self.num_rel = n_relation 
        self.users_all = torch.from_numpy(list_users).to(self.device)
        self.items_all = torch.from_numpy(list_items).to(self.device)
        self.adj_ent = adj_entity
        self.adj_rel = adj_relation
        self.user_click_item_sequence = user_click_item_pos
        self.transE_model = transE_model

        self.n_iter = args.n_iter 
        self.batch_size = args.batch_size 
        self.click_sequence_size = args.click_sequence_size
        self.dim = args.dim 
        self.n_neighbor = args.neighbor_sample_size 

        self.input_size = args.rnn_input_size
        self.hidden_size = args.rnn_hidden_size
        self.num_layers = args.rnn_num_layers

        self.pad_size = args.click_sequence_size
        self.dropout = 0.2
        self.lambd = args.lambd

        self.usr = torch.nn.Embedding(self.num_user, args.dim)
        self.ent = torch.nn.Embedding(self.num_ent, args.dim)
        self.rel = torch.nn.Embedding(self.num_rel, args.dim)

        self.gru = nn.GRU(input_size=self.dim,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               batch_first=False,
                               bidirectional=True)

        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)
        self.mha = nn.MultiheadAttention(self.dim, num_heads = 1)
        self.context = Context(self.dim, self.hidden_size, self.num_layers)
        self.transform = Transform(self.dim, self.pad_size, self.dropout, self.device)

        self.relu = nn.ReLU()
        self.criterion = nn.BCELoss()

    def _aggregate(self, user_embeddings, entities, relations, ):
        entity_vectors = []
        entity_vectors = [self.ent(entity.to(device=self.device)) for entity in entities]
        # relation_vectors = [self.rel(relation.to(device=self.device)) for relation in relations]
        relation_vectors = [out_embdding_relation(self.transE_model, relation.to(device=self.device), self.device) for relation in relations]
        # entity_vectors = [self.ent(entity.to(device=self.device)) for entity in entities]
        entity_vectors = [out_embdding_entity(self.transE_model, entity.to(device=self.device), self.device) for entity in entities]
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors = entity_vectors[hop],
                    neighbor_vectors = entity_vectors[hop + 1].view((self.click_sequence_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations = relation_vectors[hop].view((self.click_sequence_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings = user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        return entity_vectors[0].view((self.click_sequence_size, self.dim))
    
    def _get_neighbors(self, v):
        entities = [v] 
        relations = []
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]].cpu()).view((self.click_sequence_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]].cpu()).view((self.click_sequence_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations


    def train_forward(self, u, v, labels):
        batch_size = u.size(0)
        u = u.type(torch.int64)
        u = u.view((-1, 1))
        v = v.view((-1, 1))
        user_embeddings = self.usr(u).squeeze(dim=1)

        u_aggregate_item_embeddings = torch.empty(batch_size, self.click_sequence_size, self.dim, dtype=torch.float32)
        for i in range(batch_size):
            ui = u[i].item()
            ui_click_item = list(self.user_click_item_sequence[ui])
            ui_click_item = torch.from_numpy(np.array(ui_click_item)).view((-1, 1)).long() 
            entities, relations = self._get_neighbors(ui_click_item) 
            item_embeddings = self._aggregate(user_embeddings, entities, relations) 
            u_aggregate_item_embeddings[i] = item_embeddings
        u_aggregate_item_embeddings = self.relu(u_aggregate_item_embeddings)
        u_aggregate_item_embeddings = u_aggregate_item_embeddings.to(device=self.device)
        trans_out = self.transform(x=u_aggregate_item_embeddings) 

        gru_outputs, hidden_outs = self.gru(trans_out) 
        gru_outputs = self.relu(gru_outputs) 
        feat = torch.concat([gru_outputs, trans_out], dim=2)
        feat = torch.sum(feat, dim=1)
        fc = nn.Linear(in_features = feat.size(1), out_features = self.dim).to(self.device)
        feat = fc(feat) 
        feat = self.relu(feat)

        # item_embeddings_v = self.ent(v).squeeze(dim=1)
        item_embeddings_v = out_embdding_entity(self.transE_model, v, self.device).squeeze(dim=1)
        scores = (feat * item_embeddings_v).sum(dim=1)
        outs = torch.sigmoid(scores)
        bpr_loss= self.criterion(outs, labels)
        intents_distance = torch.sigmoid(gru_outputs - trans_out)
        intents_loss = self.criterion(intents_distance, torch.ones_like(intents_distance, dtype=torch.float32))
        entities, relations = self._get_neighbors(v)
        item_level_embedding = self._aggregate(user_embeddings, entities, relations)
        item_level_embedding = self.relu(item_level_embedding)
        sim_score_item = torch.mul(item_level_embedding, item_embeddings_v).sum(dim=1) 
        sim_score_item = torch.sigmoid(sim_score_item)
        pre_items_label = torch.ones_like(sim_score_item, dtype=torch.float32)
        ssl_loss_item = self.criterion(sim_score_item, pre_items_label)
        user_level_embedding = u_aggregate_item_embeddings.mean(dim=1)
        user_level_embedding = self.relu(user_level_embedding)
        sim_score_users = torch.mul(user_level_embedding, user_embeddings).sum(dim=1) 
        sim_score_users = torch.sigmoid(sim_score_users)
        pre_users_label = torch.ones_like(sim_score_users, dtype=torch.float32)
        ssl_loss_users = self.criterion(sim_score_users, pre_users_label)
        total_loss = bpr_loss + self.lambd * ssl_loss_item + self.lambd * ssl_loss_users + self.lambd * intents_loss
        total_loss = bpr_loss + self.lambd * ssl_loss_item
        return total_loss


    def forward(self, u, v):
        batch_size = u.size(0)
        u = u.type(torch.int64)
        user_embeddings = self.usr(u.view((-1, 1))).squeeze(dim=1)
        u_aggregate_item_embeddings = torch.empty(batch_size, self.click_sequence_size, self.dim, dtype=torch.float32)
        for i in range(batch_size):
            ui = u[i].item()
            ui_click_item = list(self.user_click_item_sequence[ui])
            ui_click_item = torch.from_numpy(np.array(ui_click_item)).view((-1, 1)).long() 
            entities, relations = self._get_neighbors(ui_click_item) 
            item_embeddings = self._aggregate(user_embeddings, entities, relations) 
            u_aggregate_item_embeddings[i] = item_embeddings
        u_aggregate_item_embeddings = self.relu(u_aggregate_item_embeddings)
        u_aggregate_item_embeddings = u_aggregate_item_embeddings.to(device=self.device)

        trans_out = self.transform(x=u_aggregate_item_embeddings) 
        gru_outputs, _ = self.gru(trans_out) 
        gru_outputs = self.relu(gru_outputs) 
        feat = torch.concat([gru_outputs, trans_out], dim=2)
        feat = torch.sum(feat, dim=1)
        fc = nn.Linear(in_features = feat.size(1), out_features = self.dim).to(self.device)
        feat = fc(feat) 
        feat = self.relu(feat)

        # item_embeddings_v = self.ent(v).squeeze(dim=1)
        item_embeddings_v = out_embdding_entity(self.transE_model, v, self.device).squeeze(dim=1)
        scores = (feat * item_embeddings_v).sum(dim=1) 
        outs = torch.sigmoid(scores)
        return outs
