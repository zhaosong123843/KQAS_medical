import json

import difflib

from py2neo import Graph
from slot_config import *
from build_NLU_model.predict import nlu_predict
from build_NER_model.model_predict import ner_predict

graph = Graph('neo4j://localhost:7687', auth=("neo4j", "12345678"), name="neo4j")
unrecognized_replay = unrecognized['replay_answer']


def entity_similarity(entity_type, text):
    cql = "MATCH (n:{entity_type}) RETURN n.name".format(entity_type=entity_type + 's')
    entities = graph.run(cql).data()

    max_sim_entity = ''
    max_lev_distance = 0
    for i, item in enumerate(entities):
        # 计算Levenshtein距离
        val = list(item.values())[0]
        lev_distance = difflib.SequenceMatcher(None, text, val).ratio()
        if lev_distance > 0.6 and lev_distance < 1:
            if lev_distance > max_lev_distance:
                max_lev_distance = lev_distance
                max_sim_entity = val

    return max_sim_entity


def robot_answer(text, old_entity=None, old_intent=None):
    "二次问话后的回答情景"
    if text == "是的":
        user_intent = old_intent
        entity_info = old_entity
    elif text == "不是":
        return unrecognized_replay
    else:
        # 意图识别
        user_intent = nlu_predict(text)
        user_intent = user_intent.strip(' ')
        # 实体识别
        entity_info = ner_predict(text)

    try:
        slot_info = slot_dict[user_intent]
    except KeyError:  # 没有识别到意图
        return unrecognized_replay

    cql_template = slot_info.get('cql_template')
    deny_response = slot_info.get('deny_response')
    ask_template = slot_info.get('ask_template')
    reply_template = slot_info.get('reply_template')

    if not entity_info:  # 实体不存在，但是有意图，可能是二次提问的情况
        if user_intent:
            entity_info = old_entity  # 将上次提问的实体赋予本次问话中
    # print(user_intent, entity_info)
    if entity_info:
        entity_type = entity_info[0][1]
        if entity_type != 'disease':  # 目前只能根据疾病来回答问题
            return unrecognized_replay

        entity = entity_info[0][0]

        if isinstance(cql_template, list):  # 多查询语句
            res = []
            for q in cql_template:
                cql = q.format(Disease=entity)
                data = graph.run(cql).data()
                res.extend([list(item.values())[0] for item in data if list(item.values())[0] != None])
        else:  # 单查询语句
            cql = cql_template.format(Disease=entity)
            # print(cql)
            data = graph.run(cql).data()
            res = [list(item.values())[0] for item in data if list(item.values())[0] != None]

        if not res:  # 没有检索到答案
            # 检测是否存在该实体
            cql = "MATCH(p:diseases) WHERE p.name='{Disease}' RETURN p.name".format(Disease=entity)
            data = graph.run(cql).data()

            if not data:
                # 文本相似度匹配
                sim_entity = entity_similarity(entity_type, entity)
                # 二次确认相似实体
                reply = ask_template.format(Disease=sim_entity)
                entity_info[0][0] = sim_entity

                return [reply, entity_info, user_intent]

            reply = deny_response.format(Disease=entity)
        else:
            answer = "、".join([str(i) for i in res])
            reply_template = reply_template.format(Disease=entity)
            reply = reply_template + answer
        return [reply, entity_info, user_intent]
    else:
        return unrecognized_replay


if __name__ == "__main__":

    old_entity = ''
    old_intent = ''
    while True:
        text = input('请输入问句：')
        if old_entity:
            reply = robot_answer(text, old_entity, old_intent)
        else:
            reply = robot_answer(text)

        if isinstance(reply, list):
            answer = reply[0]
            old_entity = reply[1]
            old_intent = reply[2]
        else:
            answer = reply
            old_entity = ''
            old_intent = ''

        print(answer)

    # entity_similarity('疾病','肺炎')
