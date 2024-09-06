import json
import codecs
import threading

from tqdm import tqdm # 进度条
from py2neo import Graph

class BiuldMedicalGraph:

    def __init__(self):

        self.graph = Graph('neo4j://localhost:7687', auth=("neo4j", "12345678"), name="neo4j")
        self.data_path = './data/medical.json'
        # 定义实体节点
        self.drugs = [] # 药物
        self.recipes = [] # 菜谱
        self.foods = [] # 食物
        self.checks = [] # 检查
        self.departments = [] # 科室
        self.producers = [] # 药企
        self.diseases = [] # 疾病
        self.symptoms = [] # 症状
        self.diseases_infos = [] # 疾病描述

        # 构建实体间关系
        self.rels_department = []
        self.rels_noteat = []
        self.rels_doeat = []
        self.rels_recommandeat = []
        self.rels_commonddrug = []
        self.rels_recommanddrug = []
        self.rels_check = []
        self.rels_drug_producer = []
        self.rels_symptom = []
        self.rels_acompany = []
        self.rels_category = []

    def extract_triples(self):

        with open(self.data_path,'r',encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines,ncols=80):
                data_json = json.loads(line)
                disease_dict = {}
                disease = data_json['name']
                disease_dict['name'] = disease
                self.diseases.append(disease)

                disease_dict['desc'] = ''
                disease_dict['prevent'] = ''
                disease_dict['cause'] = ''
                disease_dict['easy_get'] = ''
                disease_dict['cure_department'] = ''
                disease_dict['cure_way'] = ''
                disease_dict['cure_lasttime'] = ''
                disease_dict['symptom'] = ''
                disease_dict['cured_prob'] = ''

                if 'symptom' in data_json:
                    self.symptoms += data_json['symptom']
                    for symptom in data_json['symptom']:
                        self.rels_symptom.append([disease,'has_symptom',symptom])

                if 'acompany' in data_json:
                    for acompany in data_json['acompany']:
                        self.rels_acompany.append([disease,'acompany',acompany])
                        self.diseases.append(acompany)

                if 'desc' in data_json:
                    disease_dict['desc'] = data_json['desc']

                if 'prevent' in data_json:
                    disease_dict['prevent'] = data_json['prevent']

                if 'cause' in data_json:
                    disease_dict['cause'] = data_json['cause']

                if 'get_prob' in data_json:
                    disease_dict['get_prob'] = data_json['get_prob']

                if 'easy_get' in data_json:
                    disease_dict['easy_get'] = data_json['easy_get']

                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                        self.rels_category.append([disease,'cure_department',cure_department[0]])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_department.append([small,'belongs_to',big])
                        self.rels_category.append([disease,'cure_department','small'])

                    disease_dict['cure_department'] = cure_department
                    self.departments += cure_department

                if 'cure_way' in data_json:
                    disease_dict['cure_way'] = data_json['cure_way']

                if 'cure_lasttime' in data_json:
                    disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                if 'cured_prob' in data_json:
                    disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_commonddrug.append([disease, 'has_common_drug', drug])
                    self.drugs += common_drug

                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    for drug in recommand_drug:
                        self.rels_recommanddrug.append([disease, 'recommand_drug', drug])

                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_noteat.append([disease, 'not_eat', _not])

                    self.foods += not_eat
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_doeat.append([disease, 'do_eat', _do])

                    self.foods += do_eat

                if 'recommand_eat' in data_json:
                    recommand_eat = data_json['recommand_eat']
                    for _recommand in recommand_eat:
                        self.rels_recommandeat.append([disease, 'recommand_recipes', _recommand])
                    self.recipes += recommand_eat

                if 'check' in data_json:
                    check = data_json['check']
                    for _check in check:
                        self.rels_check.append([disease, 'need_check', _check])
                    self.checks += check

                if 'drug_detail' in data_json:
                    for det in data_json['drug_detail']:
                        det_spilt = det.split('(')
                        if len(det_spilt) == 2:
                            p, d = det_spilt
                            d = d.rstrip(')')
                            if p.find(d) > 0:
                                p = p.rstrip(d)
                            self.producers.append(p)
                            self.drugs.append(d)
                            self.rels_drug_producer.append([p, 'production', d])
                        else:
                            d = det_spilt[0]
                            self.drugs.append(d)

                self.diseases_infos.append(disease_dict)

    def write_nodes(self,entity,entity_type):
        print("写入 {0} 实体".format(entity_type))
        for node in tqdm(set(entity),ncols=80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type, entity_name=node.replace("'", ""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def write_edges(self,triples,head_type,tail_type):
        # print("写入 {0} 关系".format(triples))
        for head, relation, tail in tqdm(triples, ncols=80):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
                tail=tail.replace("'", ""), relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def create_entities(self):
        self.write_nodes(self.drugs,'drugs')
        self.write_nodes(self.recipes, 'recipes')
        self.write_nodes(self.foods, 'foods')
        self.write_nodes(self.checks, 'checks')
        self.write_nodes(self.departments, 'departments')
        self.write_nodes(self.producers, 'producers')
        self.write_nodes(self.diseases, 'diseases')
        self.write_nodes(self.symptoms, 'symptoms')

    def create_relations(self):
        self.write_edges(self.rels_department,'departments','departments')
        self.write_edges(self.rels_noteat, 'diseases', 'foods')
        self.write_edges(self.rels_doeat, 'diseases', 'foods')
        self.write_edges(self.rels_recommandeat, 'diseases', 'recipes')
        self.write_edges(self.rels_commonddrug, 'diseases', 'drugs')
        self.write_edges(self.rels_recommanddrug, 'diseases', 'drugs')
        self.write_edges(self.rels_check, 'diseases', '检查')
        self.write_edges(self.rels_drug_producer, 'producers', 'drugs')
        self.write_edges(self.rels_symptom, 'diseases', 'symptoms')
        self.write_edges(self.rels_acompany, 'diseases', 'diseases')
        self.write_edges(self.rels_category, 'diseases', 'departments')

    def set_attributes(self, entity_infos, etype):
        print("写入 {0} 实体的属性".format(etype))
        for e_dict in tqdm(entity_infos[892:], ncols=80):
            name = e_dict['name']
            del e_dict['name']
            for k, v in e_dict.items():
                if k in ['cure_department', 'cure_way']:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}={v}""".format(label=etype, name=name.replace("'", ""), k=k, v=v)
                else:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}='{v}'""".format(label=etype, name=name.replace("'", ""), k=k,
                                                  v=v.replace("'", "").replace("\n", ""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)

    def set_diseases_attributes(self):
        t = threading.Thread(target=self.set_attributes, args=(self.diseases_infos, "diseases"))
        t.setDaemon(False)
        t.start()

    def export_data(self, data, path):
        if isinstance(data[0], str):
            data = sorted([d.strip("...") for d in set(data)])
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def export_entitys_relations(self):
        self.export_data(self.drugs, './data/json/drugs.json')
        self.export_data(self.recipes, './data/json/recipes.json')
        self.export_data(self.foods, './data/json/foods.json')
        self.export_data(self.checks, './data/json/checks.json')
        self.export_data(self.departments, './data/json/departments.json')
        self.export_data(self.producers, './data/json/producers.json')
        self.export_data(self.diseases, './data/json/diseases.json')
        self.export_data(self.symptoms, './data/json/symptoms.json')

        self.export_data(self.rels_department, './data/json/rels_department.json')
        self.export_data(self.rels_noteat, './data/json/rels_noteat.json')
        self.export_data(self.rels_doeat, './data/json/rels_doeat.json')
        self.export_data(self.rels_recommandeat, './data/json/rels_recommandeat.json')
        self.export_data(self.rels_commonddrug, './data/json/rels_commonddrug.json')
        self.export_data(self.rels_recommanddrug, './data/json/rels_recommanddrug.json')
        self.export_data(self.rels_check, './data/json/rels_check.json')
        self.export_data(self.rels_drug_producer, './data/json/rels_drug_producer.json')
        self.export_data(self.rels_symptom, './data/json/rels_symptom.json')
        self.export_data(self.rels_acompany, './data/json/rels_acompany.json')
        self.export_data(self.rels_category, './data/json/rels_category.json')


if __name__ == "__main__":
    bmg = BiuldMedicalGraph()
    bmg.extract_triples()
    # bmg.create_entities()
    bmg.create_relations()
    bmg.set_diseases_attributes()
    bmg.export_entitys_relations()
    # with open('./data/medical.json','r',encoding='utf-8') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         print(line)