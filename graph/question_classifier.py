import os

import ahocorasick


class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # Feature word paths
        self.disease_path = os.path.join(cur_dir, 'dict/disease.txt')
        self.department_path = os.path.join(cur_dir, 'dict/department.txt')
        self.check_path = os.path.join(cur_dir, 'dict/check.txt')
        self.drug_path = os.path.join(cur_dir, 'dict/drug.txt')
        self.food_path = os.path.join(cur_dir, 'dict/food.txt')
        self.producer_path = os.path.join(cur_dir, 'dict/producer.txt')
        self.symptom_path = os.path.join(cur_dir, 'dict/symptom.txt')
        self.deny_path = os.path.join(cur_dir, 'dict/deny.txt')

        # Load feature words
        self.disease_wds = [i.strip() for i in open(self.disease_path, encoding='utf-8') if i.strip()]
        self.department_wds = [i.strip() for i in open(self.department_path, encoding='utf-8') if i.strip()]
        self.check_wds = [i.strip() for i in open(self.check_path, encoding='utf-8') if i.strip()]
        self.drug_wds = [i.strip() for i in open(self.drug_path, encoding='utf-8') if i.strip()]
        self.food_wds = [i.strip() for i in open(self.food_path, encoding='utf-8') if i.strip()]
        self.producer_wds = [i.strip() for i in open(self.producer_path, encoding='utf-8') if i.strip()]
        self.symptom_wds = [i.strip() for i in open(self.symptom_path, encoding='utf-8') if i.strip()]

        self.region_words = set(self.department_wds + self.disease_wds + self.check_wds +
                                self.drug_wds + self.food_wds + self.producer_wds + self.symptom_wds)
        self.deny_words = [i.strip() for i in open(self.deny_path, encoding='utf-8') if i.strip()]

        # Build Aho-Corasick automaton
        self.region_tree = self.build_actree(list(self.region_words))

        # Build dictionary
        self.wdtype_dict = self.build_wdtype_dict()

        # Question words
        self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']
        self.cause_qwds = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何',
                           '如何才会', '怎么才会', '会导致', '会造成']
        self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现',
                              '伴随发生', '伴随', '共现']
        self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱',
                          '菜谱', '食用', '食物', '补品']
        self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
        self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得', '逃开', '避开',
                             '避掉', '躲开', '躲掉', '绕开',
                             '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不', '如何才能不',
                             '怎样才不', '怎么才不', '咋样才不', '咋才不', '如何才不',
                             '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不',
                             '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时',
                              '多少年']
        self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法',
                             '咋治', '怎么办', '咋办', '咋治']
        self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性',
                              '能治', '可治', '可以治', '可以医']
        self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']
        self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']
        self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']
        self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用',
                          '用处', '用途',
                          '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']

        print('Model initialization finished...')

    def classify(self, question):
        data = {}
        medical_dict = self.check_medical(question)
        if not medical_dict:
            return {}
        data['args'] = medical_dict

        # Collect entity types involved in the question
        types = []
        for type_ in medical_dict.values():
            types += type_

        question_types = []

        # Symptom-related questions
        if self.check_words(self.symptom_qwds, question):
            if 'disease' in types:
                question_types.append('disease_symptom')
            if 'symptom' in types:
                question_types.append('symptom_disease')

        # Cause-related questions
        if self.check_words(self.cause_qwds, question) and ('disease' in types):
            question_types.append('disease_cause')

        # Complication-related questions
        if self.check_words(self.acompany_qwds, question) and ('disease' in types):
            question_types.append('disease_acompany')

        # Food-related questions
        if self.check_words(self.food_qwds, question) and 'disease' in types:
            deny_status = self.check_words(self.deny_words, question)
            question_types.append('disease_not_food' if deny_status else 'disease_do_food')

        if self.check_words(self.food_qwds + self.cure_qwds, question) and 'food' in types:
            deny_status = self.check_words(self.deny_words, question)
            question_types.append('food_not_disease' if deny_status else 'food_do_disease')

        # Drug-related questions
        if self.check_words(self.drug_qwds, question) and 'disease' in types:
            question_types.append('disease_drug')

        if self.check_words(self.cure_qwds, question) and 'drug' in types:
            question_types.append('drug_disease')

        # Check-related questions
        if self.check_words(self.check_qwds, question) and 'disease' in types:
            question_types.append('disease_check')

        if self.check_words(self.check_qwds + self.cure_qwds, question) and 'check' in types:
            question_types.append('check_disease')

        # Prevention-related questions
        if self.check_words(self.prevent_qwds, question) and 'disease' in types:
            question_types.append('disease_prevent')

        # Duration-related questions
        if self.check_words(self.lasttime_qwds, question) and 'disease' in types:
            question_types.append('disease_lasttime')

        # Treatment-related questions
        if self.check_words(self.cureway_qwds, question) and 'disease' in types:
            question_types.append('disease_cureway')

        # Cure probability questions
        if self.check_words(self.cureprob_qwds, question) and 'disease' in types:
            question_types.append('disease_cureprob')

        # Susceptibility questions
        if self.check_words(self.easyget_qwds, question) and 'disease' in types:
            question_types.append('disease_easyget')

        # Fallback options
        if not question_types:
            if 'disease' in types:
                question_types.append('disease_desc')
            elif 'symptom' in types:
                question_types.append('symptom_disease')

        data['question_types'] = question_types
        return data

    def build_wdtype_dict(self):
        wd_dict = {}
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.disease_wds:
                wd_dict[wd].append('disease')
            if wd in self.department_wds:
                wd_dict[wd].append('department')
            if wd in self.check_wds:
                wd_dict[wd].append('check')
            if wd in self.drug_wds:
                wd_dict[wd].append('drug')
            if wd in self.food_wds:
                wd_dict[wd].append('food')
            if wd in self.symptom_wds:
                wd_dict[wd].append('symptom')
            if wd in self.producer_wds:
                wd_dict[wd].append('producer')
        return wd_dict

    def build_actree(self, wordlist):
        automaton = ahocorasick.Automaton()
        for idx, word in enumerate(wordlist):
            automaton.add_word(word, (idx, word))
        automaton.make_automaton()
        return automaton

    def check_medical(self, question):
        region_wds = []
        for end_index, (insert_order, original_value) in self.region_tree.iter(question):
            region_wds.append(original_value)

        # Remove overlapping words
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)

        final_wds = [i for i in region_wds if i not in stop_wds]
        final_dict = {i: self.wdtype_dict.get(i, []) for i in final_wds}

        return final_dict

    def check_words(self, wds, sent):
        return any(wd in sent for wd in wds)


if __name__ == '__main__':
    handler = QuestionClassifier()
    while True:
        try:
            question = input('Please input your question (or press Ctrl+C to exit): ')
            data = handler.classify(question)
            print(data)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
