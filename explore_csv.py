import pandas as pd
import json
import yaml
import copy
import os
from sacremoses import MosesTokenizer

tokenizer = MosesTokenizer(lang='en')


def tokenize(line):
    line = tokenizer.tokenize(line.strip(), return_str=True, escape=False)
    line = line.replace("e _ 1", "e_1")
    line = line.replace("e _ 2", "e_2")
    return line


m1_tag = "MODE1"
m2_tag = "MODE2"
l_tag = "LENGTH"

train_files = f"/data/1.2345678910_train/{m1_tag}_1.{l_tag}_train_{m2_tag}.txt"
test_files = f"/data/1.{l_tag}_test/{m1_tag}_1.{l_tag}_test_{m2_tag}.txt"

mode1 = ["no_proof", "short_proof", "long_proof"]
mode2 = ["facts", "amt", "both"]
train_lengths = ["2", "4", "6"]
test_lengths = ["3", "5", "7", "8", "9", "10"]
all_lengths = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]


def extract_first_names(lines):
    fns = set()
    for l in lines:
        # take the story
        if '<CLEAN>' in l:
            story = l.split('<CLEAN>')[1].split('<QUERY>')[0]
        else:
            story = l.split('<STORY>')[1].split('<QUERY>')[0]
        # get the sentences of the story
        for sent in story.strip().split('.'):
            sent = sent.strip()
            if len(sent) == 0: continue
            #################
            ### OPTION 1 ####
            #######################################################
            # lowercase the first char
            sent = sent[0].lower() + sent[1:]
            # all other upper case words are considered first names
            fns.update([w for w in sent.split() if w[0].isupper()])
            #######################################################
            ### OPTION 2 #
            #######################################################
            #try:
            #    rel, e1, e2 = extract_entities_and_relation(sent)
            #    fns.update([e1, e2])
            #except IndexError as e:
            #    print(f"INDEX ERROR ON THIS LINE: '{sent}'")
            #    raise e
            #######################################################
    return fns


def main1():
    for m1 in mode1:
        for m2 in mode2:
            print(f"testing {m1} + {m2}...")
            print(f"loading training files")
            train_lines = []
            for l in train_lengths:
                print(f"  reading stories of {l} step proofs...")
                fn = train_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
                with open(fn, 'r') as f:
                    train_lines.extend(f.readlines())
            print(f"number of train lines: {len(train_lines)}")

            print("extracting first names...")
            train_fns = list(extract_first_names(train_lines))
            print(f"got {len(train_fns)} unique first names in training set")
            print(f"first 10: {train_fns[:10]}")

            #####################################

            print(f"loading testing files")
            test_lines = []
            for l in test_lengths:
                print(f"  reading stories of {l} step proofs...")
                fn = test_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
                with open(fn, 'r') as f:
                    test_lines.extend(f.readlines())
            print(f"number of TOTAL test lines: {len(test_lines)}")

            print("extracting first names...")
            test_fns = list(extract_first_names(test_lines))
            print(f"got {len(test_fns)} unique first names in testing set")
            print(f"first 10: {test_fns[:10]}")

            ######################################

            print("")
            tmp = set(train_fns) - set(test_fns)
            print(f"train names - test names: {len(tmp)} = {100*len(tmp)/len(train_fns)}% of train names are unique")
            print(f"few of them: {list(tmp)[:10]}")

            print("")
            tmp = set(test_fns) - set(train_fns)
            print(f"test names - train names: {len(tmp)} = {100*len(tmp)/len(test_fns)}% of test names are unique")
            print(f"few of them: {list(tmp)[:10]}")

            print("")
            print("")


"""
def get_data(df, lengths):
    combinations = set([])  # set of unique relation combination used ex: (rel1, rel2)
    proof_steps = set([])  # set of unique proof steps of the form (rel3, A, C, rel1, A, B, rel2, B, C)
    proofs = set([])  # set of entire proofs of the form "[ {'A-rel3-C': ['A-rel1-B', 'B-rel2-C']} , {...}, ...]"
    statements = set([])  # set of unique statement of the form (rel, A, B)

    for idx, row in df.iterrows():
        # Get the task ID
        task_id = row['task_name'].replace('task_1.', '')
        if task_id in lengths:
            # Get the f_comb
            f_comb = row['f_comb']
            combinations.add(f_comb)
            # Get the proof state
            proof_raw = row['proof_state'].replace('), (', ',').replace('(', '').replace(')', '').replace('\', \'', '--').replace('\'', '\"')
            proofs.add(proof_raw)
            proof_state = json.loads(proof_raw)
            # ex: [
            #     {"Sara-sister-Kathryn": [
            #         "Sara-father-John",
            #         "John-daughter-Kathryn"
            #     ]},
            #     {"Sara-father-John": [
            #         "Sara-daughter-Kristie",
            #         "Kristie-grandfather-John"
            #     ]
            # }]
            for rule in proof_state:
                for conclusion, predicates in rule.items():
                    statements.add(conclusion)
                    statements.update(predicates)
                    proof_steps.add((conclusion, predicates[0], predicates[1]))
    return combinations, proof_steps, proofs, statements


def main2():
    print(f"Loading training proofs...")

    # df = pd.read_csv("/data/1.2345678910_train/1.2345678910_train.csv")
    df = pd.read_csv("/data/1.2345678910_train/1.2345678910_train.csv")
    print(f"#of lines: {len(df)}")

    train_combinations, train_proof_steps, train_proofs, train_statements = get_data(df, train_lengths)
    print(f"#of train combinations: {len(train_combinations)}")
    print(f"#of train proof steps: {len(train_proof_steps)}")
    print(f"#of train proofs: {len(train_proofs)}")
    print(f"#of train statements: {len(train_statements)}")

    for l in all_lengths:
        print("")
        print(f"Loading test proofs of length {l}...")
        # df = pd.read_csv(f"/data/1.{l}_test/1.{l}_test.csv")
        df = pd.read_csv(f"/data/1.{l}_test/1.{l}_test.csv")
        print(f"#of lines: {len(df)}")

        test_combinations, test_proof_steps, test_proofs, test_statements = get_data(df, [l])
        print(f"#of test combinations: {len(test_combinations)}")
        print(f"#of test proof steps: {len(test_proof_steps)}")
        print(f"#of test proofs: {len(test_proofs)}")
        print(f"#of test statements: {len(test_statements)}")

        tmp = train_combinations - test_combinations
        print("")
        print(f"train - test combinations: {len(tmp)} = {100*len(tmp) / len(train_combinations)}% of train combinations are unique")
        print(f"few of them: {list(tmp)[:5]}")
        tmp = test_combinations - train_combinations
        print("")
        print(f"test - train combinations: {len(tmp)} = {100 * len(tmp) / len(test_combinations)}% of test combinations are unique")
        print(f"few of them: {list(tmp)[:5]}")

        tmp = train_proof_steps - test_proof_steps
        print("")
        print(f"train - test proof steps: {len(tmp)} = {100 * len(tmp) / len(train_proof_steps)}% of train proof steps are unique")
        print(f"few of them: {list(tmp)[:5]}")
        tmp = test_proof_steps - train_proof_steps
        print("")
        print(f"test - train proof steps: {len(tmp)} = {100 * len(tmp) / len(test_proof_steps)}% of test proof steps are unique")
        print(f"few of them: {list(tmp)[:5]}")

        '''
        tmp = train_proofs - test_proofs
        print("")
        print(f"train - test proofs: {len(tmp)} = {100 * len(tmp) / len(train_proofs)}% of train proofs are unique")
        if len(tmp) > 0: print(f"one of them: {list(tmp)[0]}")
        tmp = test_proofs - train_proofs
        print("")
        print(f"test - train proofs: {len(tmp)} = {100 * len(tmp) / len(test_proofs)}% of test proofs are unique")
        if len(tmp) > 0: print(f"one of them: {list(tmp)[0]}")
        '''

        tmp = train_statements - test_statements
        print("")
        print(f"train - test statements: {len(tmp)} = {100 * len(tmp) / len(train_statements)}% of train statements are unique")
        print(f"few of them: {list(tmp)[:5]}")
        tmp = test_statements - train_statements
        print("")
        print(f"test - train statements: {len(tmp)} = {100 * len(tmp) / len(test_statements)}% of test statements are unique")
        print(f"few of them: {list(tmp)[:5]}")

        print("")
"""


def custom_add(dictionary, key, element):
    try:
        dictionary[key].add(element)
    except KeyError:
        dictionary[key] = {element}


def size_of(dictionary, keys):
    s = set()
    for k in keys:
        s.update(dictionary[k])
    return len(s)


def extract_entities_and_relation(line):
    line = line.replace(',', '').strip()

    # extract relation from line
    rel = [w for w in line.split() if w in rel_to_phrases]
    if len(rel) < 1:
        print(f"no relation in line '{line}'")
        raise IndexError()
    elif len(rel) > 1:
        print(f"more than one relation in line '{line}'")
        raise IndexError()
    else:
        rel = rel[0]

    # print(f"relation {rel} found in line {line}")

    # extract who is e_1 and who is e_2
    e1, e2 = None, None
    first_names = list(set([w for w in line.replace('The ', 'the ').split() if w[0].isupper()]))
    if len(first_names) != 2:
        print(f"line '{line}' does not have exactly 2 names: {first_names}")
        raise IndexError()
    for p in rel_to_phrases[rel]:
        # either ( e_1=fn[0] and e_2=fn[1] ) or ( e_1=fn[1] and e_2=fn[0] )
        p1 = p.replace("e_1", first_names[0]).replace("e_2", first_names[1])
        p2 = p.replace("e_1", first_names[1]).replace("e_2", first_names[0])
        # print(f"[line] '{line}'")
        # print(f"[p1]   '{p1}'")
        # print(f"[p2]   '{p2}'")
        # print("")
        if line == p1:
            e1 = first_names[0]
            e2 = first_names[1]
            break
        elif line == p2:
            e1 = first_names[1]
            e2 = first_names[0]
            break

    if None in (e1, e2):
        print(f"could not find entities in '{line}'")
        for p in rel_to_phrases[rel]:
            p1 = p.replace("e_1", first_names[0]).replace("e_2", first_names[1])
            p2 = p.replace("e_1", first_names[1]).replace("e_2", first_names[0])
            print(f"[p1]   '{p1}'")
            print(f"[p2]   '{p2}'")
            print("")
        raise IndexError()

    return rel, e1, e2


def get_data(lines, source='proof'):
    """
    :param lines: array of example lines containing a story, question, proof and answer
    :param source: where to look in the line to extract data (either 'proof' or 'story')
    :return: mappings from unique proof|proof_steps|statements|first_names|relations to list of line ID with that
    """
    proofs = {}  # dict of entire proofs of the form "since <> and <> then <> . since <> and <> then <> . ..."
    proof_steps = {}  # dict of unique proof steps of the form "since A r B and B r C then A r C ."
    statements = {}  # dict of unique statement of the form "A r B"
    first_names = {}  # dict of unique first names
    relations = {}  # dict of unique relations

    for idx, line in enumerate(lines):

        if source == 'proof':
            # Get the proof
            raw_info = line.split('<PROOF>')[1].split('<ANSWER>')[0].replace('Since ', 'since ')
            proof_str = []
        elif source == 'story':
            # Get the story
            raw_info = line.split('<STORY>')[1].split('<QUERY>')[0]
            proof_str = ["none"]  # assume no proof if source = story
        else:
            raise ValueError(f"Invalid parameter source={source}.")

        for sent in raw_info.split('.'):
            if len(sent.strip()) == 0: continue
            try:
                if source == 'proof':
                    predicate1 = sent.split(' since ')[1].split(' and ')[0]
                    rel1, e11, e12 = extract_entities_and_relation(predicate1)
                    custom_add(first_names, e11, idx)
                    custom_add(first_names, e12, idx)
                    custom_add(relations, rel1, idx)
                    predicate1 = f"{e12}-{rel1}-{e11}"
                    custom_add(statements, predicate1, idx)

                    predicate2 = sent.split(' and ')[1].split(' then ')[0]
                    rel2, e21, e22 = extract_entities_and_relation(predicate2)
                    custom_add(first_names, e21, idx)
                    custom_add(first_names, e22, idx)
                    custom_add(relations, rel2, idx)
                    predicate2 = f"{e22}-{rel2}-{e21}"
                    custom_add(statements, predicate2, idx)

                    conclusion = sent.split(' then ')[1]
                    rel3, e31, e32 = extract_entities_and_relation(conclusion)
                    custom_add(first_names, e31, idx)
                    custom_add(first_names, e32, idx)
                    custom_add(relations, rel3, idx)
                    conclusion = f"{e32}-{rel3}-{e31}"
                    custom_add(statements, conclusion, idx)
                else:
                    # TODO: considering only the story will potentially skip
                    #  new relations present only in the PROOF and ANSWER
                    e12, rel1, e11, e22, rel2, e21 = None, None, None, None, None, None
                    rel3, e31, e32 = extract_entities_and_relation(sent.strip())
                    custom_add(first_names, e31, idx)
                    custom_add(first_names, e32, idx)
                    custom_add(relations, rel3, idx)

            except IndexError as e:
                print(f"INDEX ERROR ON THIS LINE: '{sent}'")
                raise e

            if source == 'proof':
                sent = f"{e12}-{rel1}-{e11} + {e22}-{rel2}-{e21} = {e32}-{rel3}-{e31}"
                custom_add(proof_steps, sent, idx)
                proof_str.append(sent)  # add this step to the proof

        proof_str = '. '.join(proof_str)
        custom_add(proofs, proof_str, idx)

    return proofs, proof_steps, statements, first_names, relations


def main3():
    m1 = "no_proof"  # short_proof | long_proof | no_proof
    m2 = "facts"  # facts | amt | both

    print(f"Loading training proofs")
    train_lines = []
    for l in train_lengths:
        print(f"  reading stories of {l} step proofs...")
        fn = train_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
        with open(fn, 'r') as f:
            train_lines.extend(f.readlines())
    print(f"number of train lines: {len(train_lines)}")

    if m1 == 'no_proof':
        train_proofs, train_proof_steps, train_statements, train_first_names, train_relations = get_data(train_lines, source='story')
    else:
        train_proofs, train_proof_steps, train_statements, train_first_names, train_relations = get_data(train_lines)
    print(f"#of train proofs: {len(train_proofs)}")
    print(f"#of train proof steps: {len(train_proof_steps)}")
    print(f"#of train statements: {len(train_statements)}")
    print(f"#of train first names: {len(train_first_names)}")
    print(f"#of train relations: {len(train_relations)}")

    for l in all_lengths:
        print("")
        print(f"Loading test proofs of length {l}...")
        fn = test_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
        if os.path.isfile(fn.replace('.txt', '_PROPER.txt')):
            fn = fn.replace('.txt', '_PROPER.txt')
            print("-- PROPER version --")
        with open(fn, 'r') as f:
            test_lines = f.readlines()
        print(f"#of lines: {len(test_lines)}")

        if m1 == 'no_proof':
            test_proofs, test_proof_steps, test_statements, test_first_names, test_relations = get_data(test_lines, source='story')
        else:
            test_proofs, test_proof_steps, test_statements, test_first_names, test_relations = get_data(test_lines)
        print(f"#of test proofs: {len(test_proofs)}")
        print(f"#of test proof steps: {len(test_proof_steps)}")
        print(f"#of test statements: {len(test_statements)}")
        print(f"#of test first names: {len(test_first_names)}")
        print(f"#of test relations: {len(test_relations)}")

        #tmp = train_proofs - test_proofs
        #print("")
        #print(f"train - test proofs: {len(tmp)} = {100 * len(tmp) / len(train_proofs)}% of train proofs are unique")
        #if len(tmp) > 0: print(f"one of them: {list(tmp)[0]}")
        tmp = test_proofs.keys() - train_proofs.keys()
        print("")
        print(f"test - train proofs: {len(tmp)} = {100 * len(tmp) / len(test_proofs)}% of test proofs are unique")
        if len(tmp) > 0: print(f"one of them: {list(tmp)[0]}")
        s = size_of(test_proofs, tmp)
        print(f"which represents {s} / {len(test_lines)} = {100 * s / len(test_lines)}% of test examples")

        if m1 != 'no_proof':
            #tmp = train_proof_steps - test_proof_steps
            #print("")
            #print(f"train - test proof steps: {len(tmp)} = {100 * len(tmp) / len(train_proof_steps)}% of train proof steps are unique")
            #print(f"few of them: {list(tmp)[:5]}")
            tmp = test_proof_steps.keys() - train_proof_steps.keys()
            print("")
            print(f"test - train proof steps: {len(tmp)} = {100 * len(tmp) / len(test_proof_steps)}% of test proof steps are unique")
            print(f"few of them: {list(tmp)[:5]}")
            s = size_of(test_proof_steps, tmp)
            print(f"which represents {s} / {len(test_lines)} = {100 * s / len(test_lines)}% of test examples")

            #tmp = train_statements - test_statements
            #print("")
            #print(f"train - test statements: {len(tmp)} = {100 * len(tmp) / len(train_statements)}% of train statements are unique")
            #print(f"few of them: {list(tmp)[:5]}")
            tmp = test_statements.keys() - train_statements.keys()
            print("")
            print(f"test - train statements: {len(tmp)} = {100 * len(tmp) / len(test_statements)}% of test statements are unique")
            print(f"few of them: {list(tmp)[:5]}")
            s = size_of(test_statements, tmp)
            print(f"which represents {s} / {len(test_lines)} = {100 * s / len(test_lines)}% of test examples")

        #tmp = train_first_names - test_first_names
        #print("")
        #print(f"train - test entities: {len(tmp)} = {100 * len(tmp) / len(train_first_names)}% of train entities are unique")
        #print(f"few of them: {list(tmp)[:5]}")
        tmp = test_first_names.keys() - train_first_names.keys()
        print("")
        print(f"test - train entities: {len(tmp)} = {100 * len(tmp) / len(test_first_names)}% of test entities are unique")
        print(f"few of them: {list(tmp)[:5]}")
        s = size_of(test_first_names, tmp)
        print(f"which represents {s} / {len(test_lines)} = {100 * s / len(test_lines)}% of test examples")

        #tmp = train_relations - test_relations
        #print("")
        #print(f"train - test relations: {len(tmp)} = {100 * len(tmp) / len(train_relations)}% of train relations are unique")
        #print(f"few of them: {list(tmp)[:5]}")
        tmp = test_relations.keys() - train_relations.keys()
        print("")
        print(f"test - train relations: {len(tmp)} = {100 * len(tmp) / len(test_relations)}% of test relations are unique")
        print(f"few of them: {list(tmp)[:5]}")
        s = size_of(test_relations, tmp)
        print(f"which represents {s} / {len(test_lines)} = {100 * s / len(test_lines)}% of test examples")

        print("")


def get_relation_mappings():
    """
    :param relations: yaml dict of relation to partial phrases
    :return: a mapping from relation to all possible phrases
    """
    rel2phrases = {}
    for key, val in relations.items():
        for gender in ['male', 'female']:
            rel = val[gender]['rel']
            rel2phrases[rel] = []
            for p in val[gender]['p']:
                rel2phrases[rel].append(tokenize(p))

    # extra hack for neice -vs- niece
    # accept both syntaxes due to vocab error
    rel2phrases['neice'] = copy.copy(rel2phrases['niece'])
    for idx, phrase in enumerate(rel2phrases['neice']):
        rel2phrases['neice'][idx] = phrase.replace('niece', 'neice')

    return rel2phrases


if __name__ == "__main__":
    with open("clutrr/store/relations_store.yaml", 'r') as f:
        relations = yaml.safe_load(f)
    rel_to_phrases = get_relation_mappings()
    del relations
    print("done.")

    # main1()
    main3()
