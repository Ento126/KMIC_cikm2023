import sys
import os
import argparse
import torch
from transE import model as model_definition
from transE import data
import torch.optim as optim




parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate value.")
parser.add_argument("--dataset_path", type=str, default="./transE/synth_data/music", help="Path to dataset.")
parser.add_argument("--vector_length", type=int, default=16, help="Length of entity/relation vector.")
parser.add_argument("--margin", type=float, default=1.0, help="Margin value in margin-based ranking loss.")
parser.add_argument("--norm", type=int, default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
parser.add_argument("--use_gpu", type=bool, default=True, help="Flag enabling gpu usage.")
args = parser.parse_args()


path = args.dataset_path
train_path = os.path.join(path, "train.txt")
entity2id, relation2id = data.create_mappings(train_path)

device = torch.device('cuda') if args.use_gpu else torch.device('cpu')



model_transE = model_definition.TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=args.vector_length,
                                    margin=args.margin,
                                    device=device, 
                                    norm=args.norm)  # type: torch.nn.Module
model_transE = model_transE.to(device=device)
optimizer = optim.SGD(model_transE.parameters(), lr=args.learning_rate)


def load_transE_model(device, model):
    # transE_model = torch.load('/data/webGroup/pengyingtao/codes/knowledge_RS/krec-extend/transE/transE_model.pth').to(device)  # full saved, path issue
    # best_model = model.to(device)
    # model.load_state_dict('transE_parmer_model.pth')
    # model= model.to(device)

    # checkpoint = torch.load('transE_parmer_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model= model.to(device)

    model.load_state_dict(torch.load(os.path.join(path, "transE_parmer_model.pth")))
    model= model.to(device)
    return model


def out_embdding_entity(model, input_entity, device):
    input_entity = input_entity.to(device)
    emb_entity = model.embdding_entity(input_entity)
    return emb_entity


def out_embdding_relation(model, input_relation, device):
    input_relation = input_relation.to(device)
    emb_relation = model.embdding_relation(input_relation)
    return emb_relation


# if __name__ == '__main__':

    # test_tensor = torch.tensor([7074,   52,  281, 7075, 1547, 1028,   35,  381,  258, 7076, 1548,    7,
    #        1,    1, 7077, 7078,  382,    0,    0, 2017,   72, 2019,  282,   73,
    #      283, 7079,  350,   47, 1549, 7080, 2023,   15,   95,  233,  139,  139,
    #      191,    2, 2031,   49, 7081,  848,  582, 7082,   27,   16, 2037, 2039,
    #      110,  383, 7083,  384, 2043,   31,  284,  121,  234,    6, 1552,  209,
    #     7084, 1553,  210,  385,  259,   34,   18, 1554, 1555,  130, 1036,   36,
    #      386,  851,    3,  316, 1556,  122,   10,   35,  258, 7085,    7,    1,
    #     7086,  583,  235,    0,    0,  131, 7087,  111,  169,   82,  432,  182,
    #       72,  497,  433,   47, 7088, 7089,  584, 2072,  852, 2073, 7090,  499,
    #      285,  388,   24,  261,  585,   12,  681, 7091,  184,  262, 2083,  853,
    #     7092,  211, 1560, 1042,  234, 7093,  142,  501])
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # transE_model = load_transE_model(device, model=model)
    
    # out = out_embdding_entity(transE_model, test_tensor, device)

    # print(out)
    
    # print('Done.')
        