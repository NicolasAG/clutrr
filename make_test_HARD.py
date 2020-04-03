import pandas as pd
import json
import yaml
import copy
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


DEBUG = False


def get_all_first_names():
    """
    :return: mapping from gender to list of ( names and binary flag to know if already used in an example )
    ex:
     {'male': {name: T/F, name: T/F, ...},
     'female': {name: T/F, name: T/F, ...}}
    """
    fns = {'male': {}, 'female': {}}

    train_fname = "/data/1.2345678910_train/1.2345678910_train.csv"
    valid_fname = "/data/1.2345678910_valid/1.2345678910_valid.csv"
    test_fnames = [
        f"/data/1.{l}_test/1.{l}_test.csv"
        for l in all_lengths
    ]

    for f_name in [train_fname, valid_fname] + test_fnames:
        df = pd.read_csv(f_name)
        for idx, row in df.iterrows():
            # Get the gender of all characters in the story
            # row['genders'] example: "Kenneth:male,Amy:female,Gail:female"
            genders_list = row['genders'].split(',')
            for person in genders_list:
                name = person.split(':')[0]
                sex = person.split(':')[1]
                fns[sex][name] = False
    return fns


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
    first_names = list(set([w for w in line.split() if w[0].isupper()]))
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


def get_data(lines):
    proofs = set([])  # set of entire proofs of the form "since <> and <> then <> . since <> and <> then <> . ..."
    proof_steps = set([])  # set of unique proof steps of the form "since A r B and B r C then A r C ."
    statements = set([])  # set of unique statement of the form "A r B"
    first_names = set([])  # set of unique first names
    relations = set([])  # set of unique relations

    for idx, line in enumerate(lines):

        # Get the proof
        proof_raw = line.split('<PROOF>')[1].split('<ANSWER>')[0].replace('Since ', 'since ')
        proof_str = []

        for sent in proof_raw.split('.'):
            if len(sent.strip()) == 0: continue
            try:
                predicate1 = sent.split(' since ')[1].split(' and ')[0]
                rel1, e11, e12 = extract_entities_and_relation(predicate1)
                first_names.update([e11, e12])
                relations.add(rel1)
                predicate1 = f"{e12}-{rel1}-{e11}"
                statements.add(predicate1)

                predicate2 = sent.split(' and ')[1].split(' then ')[0]
                rel2, e21, e22 = extract_entities_and_relation(predicate2)
                first_names.update([e21, e22])
                relations.add(rel2)
                predicate2 = f"{e22}-{rel2}-{e21}"
                statements.add(predicate2)

                conclusion = sent.split(' then ')[1]
                rel3, e31, e32 = extract_entities_and_relation(conclusion)
                first_names.update([e31, e32])
                relations.add(rel3)
                conclusion = f"{e32}-{rel3}-{e31}"
                statements.add(conclusion)
            except IndexError as e:
                print(f"INDEX ERROR ON THIS LINE: '{sent}'")
                raise e

            sent = f"{e12}-{rel1}-{e11} + {e22}-{rel2}-{e21} = {e32}-{rel3}-{e31}"
            proof_steps.add(sent)

            proof_str.append(sent)  # add this step to the proof
        proof_str = '. '.join(proof_str)
        proofs.add(proof_str)

    return proof_steps, proofs, statements, first_names, relations


def get_gender(name):
    if name in all_first_names['male']:
        return 'male'
    elif name in all_first_names['female']:
        return 'female'
    else:
        raise ValueError(f"Unknown first name: {name}")


def main3():
    m1 = "short_proof"
    m2 = "facts"

    train_first_names = set([])
    print(f"Loading training stories")
    for l in train_lengths:
        print(f"  reading stories of {l} step proofs...")
        fn = train_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
        with open(fn, 'r') as f:
            train_lines = f.readlines()
        _, _, _, first_names, _ = get_data(train_lines)
        train_first_names.update(first_names)
    print(f"#of train first names: {len(train_first_names)}")
    # make sure all names exist
    for n in train_first_names:
        assert n in all_first_names['male'] or n in all_first_names['female'], f"Unknown name from train set: {n}"

    all_test_first_names = set([])
    print(f"Loading testing stories")
    for l in all_lengths:
        print(f"  reading stories of {l} step proofs...")
        fn = test_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
        with open(fn, 'r') as f:
            test_lines = f.readlines()
        _, _, _, test_first_names, _ = get_data(test_lines)
        all_test_first_names.update(test_first_names)
    print(f"#of test first names: {len(all_test_first_names)}")
    # make sure all names exist
    for n in all_test_first_names:
        assert n in all_first_names['male'] or n in all_first_names['female'], f"Unknown name from test set: {n}"

    new_first_names = all_test_first_names - train_first_names
    print(f"#of test names not present in train names: {len(new_first_names)}")
    # make sure all names exist
    for n in new_first_names:
        assert n in all_first_names['male'] or n in all_first_names['female'], f"Unknown name from unique set: {n}"

    for l in train_lengths:
        print("")
        print(f"Loading test proofs of length {l}...")
        if DEBUG:
            fn = f"/data/1.{l}_test/tmp.txt"
        else:
            fn = test_files.replace(m1_tag, m1).replace(m2_tag, m2).replace(l_tag, l)
        with open(fn, 'r') as f:
            test_lines = f.readlines()
        print(f"#of test lines: {len(test_lines)}")

        if DEBUG:
            fn2 = None
            query_lines = None
        else:
            # also load queries line and replace those!
            fn2 = fn.replace(m1, 'queries')
            with open(fn2, 'r') as f:
                query_lines = f.readlines()
            print(f"#of query lines: {len(query_lines)}")
            assert len(query_lines) == len(test_lines)

        # replace names by names from new_first_names
        for idx, line in enumerate(test_lines):  # for all test line
            if len(line.strip()) == 0: continue  # skip empty lines

            # get the names in this line
            _, _, _, test_line_fns, _ = get_data([line])

            # find the names to replace & to keep
            names_to_keep = []
            for name in test_line_fns:
                if name in new_first_names:
                    names_to_keep.append(name)
                    gender = get_gender(name)
                    assert name in all_first_names[gender], f"unknown {gender} name: {name}"
                    all_first_names[gender][name] = True  # flag that name as being not available anymore

            # find the old_name --> new_name replacement
            replacement = {}
            for old_name in test_line_fns:
                gender = get_gender(old_name)
                # if the old name is to be kept, keep it.
                if old_name in names_to_keep:
                    new_name = old_name
                else:
                    available_names = [n for n, used in all_first_names[gender].items() if not used and n in new_first_names]
                    # if no others, keep the old name and reset the list for next round
                    if len(available_names) == 0:
                        new_name = old_name
                        if DEBUG: print(f"--keep the same name for now ({old_name}) but reset {gender} names for future use!")
                        # reset the first names for next round!
                        for n in all_first_names[gender]:
                            all_first_names[gender][n] = False
                    else:
                        new_name = available_names[0]
                replacement[old_name] = new_name
                # make sure that names exist
                assert new_name in all_first_names[gender], f"Unknown {gender} new name: {new_name}!"
                all_first_names[gender][new_name] = True

            if DEBUG: print(f"old line: {line.strip()}")
            # replace the names in the line
            for old_name, new_name in replacement.items():
                line = line.replace(' '+old_name+' ', ' '+new_name+' ')
            test_lines[idx] = line
            if DEBUG: print(f"new line: {line.strip()}")
            if DEBUG: print(f"name replacement: {replacement}")
            if DEBUG: print("")

            # also replace query lines!
            if query_lines:
                q_line = query_lines[idx]
                for old_name, new_name in replacement.items():
                    q_line = q_line.replace(' '+old_name+' ', ' '+new_name+' ')
                query_lines[idx] = q_line

        # get all new test names
        _, _, _, test_line_fns, _ = get_data(test_lines)
        # make sure all names exist
        for n in test_line_fns:
            assert n in all_first_names['male'] or n in all_first_names['female'], f"Unknown (according to finals) name from test set: {n}"

        # Save the new lines to file
        fn = fn.replace('.txt', '_HARD.txt')
        with open(fn, 'w') as f:
            f.writelines(test_lines)
        # Also save the query lines!
        if fn2 and query_lines:
            fn2 = fn2.replace('.txt', '_HARD.txt')
            with open(fn2, 'w') as f:
                f.writelines(query_lines)


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
    print("Getting relation-to-phrase mapping...")
    with open("clutrr/store/relations_store.yaml", 'r') as f:
        relations = yaml.safe_load(f)
    rel_to_phrases = get_relation_mappings()
    del relations
    print("done.")

    print(f"getting all first names...")
    all_first_names = get_all_first_names()
    print(f"got {len(all_first_names['male'])} males and {len(all_first_names['female'])} females.")

    main3()
