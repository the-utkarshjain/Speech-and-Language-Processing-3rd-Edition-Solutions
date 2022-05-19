from copy import deepcopy
import itertools

def extract_rules(grammar):
    """
    Given a string of rules, this function returns all the individual rules and the lexical rules.
    Assumption: Use '|' only for lexical rules. If a non-terminal has multiple rules, mention them seperately in seperate lines.
    Example for assumption:
    
    Correct input:
    S -> NP VP
    S -> Aux NP VP
    S -> VP

    Incorrect input:
    S -> NP VP | Aux NP VP | VP
    """
    raw_grammar = list()
    lexical_rules = list()
    non_terminals = set()

    rules = grammar.split("\n")
    for rule in rules:
        left, right = rule.strip().split("->")
        non_terminals.add(left.strip())

        left = left.strip()
        if '|' in right:
            right = right.strip().split(" | ")
            right = [r.strip() for r in right]
            lexical_rules.append((left, right))
        else:
            right = right.strip().split(" ")
            right = [r.strip() for r in right]
            raw_grammar.append((left, right))

    for left, right in raw_grammar:
        if len(right) == 1 and (right[0] not in non_terminals):
            raw_grammar.remove((left, right))
            lexical_rules.append((left, right))

    return raw_grammar, lexical_rules

def get_non_terminals_and_terminals(rules):
    all_symbols = set()
    terminals = set()
    non_terminals = set()

    for left, right in rules:
        all_symbols.add(left)
        for r in right:
            all_symbols.add(r)

        non_terminals.add(left)
    
    terminals = all_symbols - non_terminals

    return non_terminals, terminals

def get_unit_productions(grammar, lexical_rules):
    """
    Return a dictionary of all the unit productions in the grammar and lexical rules. Assumes that every unit production will give rise to another unit production or a terimal. 
    """

    unit_dict = {}
    for left, right in grammar:
        if len(right) == 1:
            unit_dict[left] = right 
    
    for left, right in lexical_rules:
        unit_dict[left] = right
    
    return unit_dict

def convert_to_CNF(raw_grammar, lexical_rules, non_terminals, terminals):
    new_grammar = deepcopy(raw_grammar)

    # Step 1: Deal with rules that mix terminals and non-terminals. Replace terminals with dummy non-terminals and add a new rule.
    for left, right in raw_grammar:
        new_grammar.remove((left, right))

        for i in range(len(right)):
            if right[i] in terminals:
                lexical_rules.append((right[i].upper(), [right[i]]))
                right[i] = right[i].upper()

        new_grammar.append((left, right))

    raw_grammar = new_grammar
    new_grammar = deepcopy(raw_grammar)

    # Step 2: Rules with right-hand sides longer than 2 are normalized through the introduction of new non-terminals that spread the longer sequences over several new rules.
    for left, right in raw_grammar:
        new_grammar.remove((left, right))

        while len(right) > 2:
            new_LHS = right[0] + '-' + right[1]
            new_grammar.append((new_LHS, [right[0], right[1]]))
            right = [new_LHS] + right[2:]
        
        new_grammar.append((left, right))

    raw_grammar = new_grammar
    new_grammar = deepcopy(raw_grammar)

    # Step 3: Eliminate unit productions by rewriting the right-hand side of the original rules with the right-hand side of all the non-unit production rules that they ultimately lead to.

    # Adding all the non-unit productions from the updated raw_grammar
    for left, right in raw_grammar:
        if len(right) == 1:
            for l, r in raw_grammar:
                if l == right[0] and len(r) == 2:
                    new_grammar.append((left, r))

    # Adding all the lexical productions from the lexical rules
    unit_dict = get_unit_productions(new_grammar, lexical_rules)

    for left, right in raw_grammar:
        edit = -1
        new_grammar.remove((left, right))
        
        while len(right) == 1:
            edit = 1
            if right[0] in unit_dict:
                right = unit_dict[right[0]]
            else:
                edit = 0
                break

        if edit == 1:
            lexical_rules.append((left, right))
        elif edit == -1:
            new_grammar.append((left, right))
        else:
            pass

    return new_grammar, lexical_rules

def display_grammar(grammar, lexical_rules):
    for left, right in grammar:
        right = " ".join(right)
        print(left + " -> " + right)
    
    for left, right in lexical_rules:
        right = " | ".join(right)
        print(left + " -> " + right)

def reverse_rules_dictionary(grammar, lexical_rules):
    reverse_rules = dict()

    for left, right in grammar:
        right = tuple(right)
        if right in reverse_rules:
            reverse_rules[right].append(left)
        else:
            reverse_rules[right] = [left]
    
    for left, right in lexical_rules:
        for r in right:
            if r in reverse_rules:
                reverse_rules[r].append(left)
            else:
                reverse_rules[r] = [left]
    
    return reverse_rules

def CKY_Parse(sentence, reverse_rules):
    sentence = sentence.split(" ")

    table = dict()
    for i in range(len(sentence) + 1):
        for j in range(i+1, len(sentence) + 1):
            table[i,j] = []
    
    for j in range(1, len(sentence) + 1):
        table[j-1, j] += reverse_rules[sentence[j-1]]

        for i in range(j-2, -1, -1):
            for k in range(i+1 , j):
                B = table[i, k]
                C = table[k, j]
                for element in itertools.product(B, C):
                    if element in reverse_rules:
                        table[i, j] += reverse_rules[element]

    return table