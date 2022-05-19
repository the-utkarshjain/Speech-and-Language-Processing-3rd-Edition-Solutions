"""13.1.py
Solution for 13.1, where we are asked to write a code to convert a grammer to CNF. I have assumed there are no epislon-rules.
"""
from utils import extract_rules, get_non_terminals_and_terminals, convert_to_CNF, display_grammar

grammar = '''\
S -> NP VP
S -> Aux NP VP
S -> VP
NP -> Pronoun
NP -> Proper-Noun
NP -> Det Nominal
Nominal -> Noun
Nominal -> Nominal Noun
Nominal -> Nominal PP
VP -> Verb
VP -> Verb NP
VP -> Verb NP PP
VP -> Verb PP
VP -> VP PP
PP -> Preposition NP
Det -> that | this | the | a
Noun -> book | flight | meal | money
Verb -> book | include | prefer
Pronoun -> I | she | me
Proper-Noun -> Houston | NWA
Aux -> does
Preposition -> from | to | on | near | through\
'''

raw_grammar, lexical_rules = extract_rules(grammar)
non_terminals, terminals = get_non_terminals_and_terminals(raw_grammar + lexical_rules)
new_grammar, lexical_rules = convert_to_CNF(raw_grammar, lexical_rules, non_terminals, terminals)
display_grammar(new_grammar, lexical_rules)