 Kelsey Wanket
 12/17/2020
 
 
 
 
 Using PCFG's to Find Probabilities of Mismatched Cases inside Coordination Phrases 
 
 
 
 
 My research on this topic so far has included using theory to find reasons for 
which we get sentences such as, "You and me are going to the store," and, "In the 
end, it's him and I," where nominative and accusative cases are mixmatched within 
coordinate phrases including two or more pronouns.
 In this project, I would like to use Python and NLTK to induce PCFG's from parsed 
corpuses, and test them on various pronoun coordinations to get the probabilities 
of each one. The phrases, "It is me/ It is I" are also important to my research, so 
that will also be used.


```python
import nltk
from nltk.corpus import treebank
parsed_sents = treebank.parsed_sents()[:]
print(parsed_sents) #print to make sure it worked
```

    [Tree('S', [Tree('NP-SBJ', [Tree('NP', [Tree('NNP', ['Pierre']), Tree('NNP', ['Vinken'])]), Tree(',', [',']), Tree('ADJP', [Tree('NP', [Tree('CD', ['61']), Tree('NNS', ['years'])]), Tree('JJ', ['old'])]), Tree(',', [','])]), Tree('VP', [Tree('MD', ['will']), Tree('VP', [Tree('VB', ['join']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['board'])]), Tree('PP-CLR', [Tree('IN', ['as']), Tree('NP', [Tree('DT', ['a']), Tree('JJ', ['nonexecutive']), Tree('NN', ['director'])])]), Tree('NP-TMP', [Tree('NNP', ['Nov.']), Tree('CD', ['29'])])])]), Tree('.', ['.'])]), Tree('S', [Tree('NP-SBJ', [Tree('NNP', ['Mr.']), Tree('NNP', ['Vinken'])]), Tree('VP', [Tree('VBZ', ['is']), Tree('NP-PRD', [Tree('NP', [Tree('NN', ['chairman'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NP', [Tree('NNP', ['Elsevier']), Tree('NNP', ['N.V.'])]), Tree(',', [',']), Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['Dutch']), Tree('VBG', ['publishing']), Tree('NN', ['group'])])])])])]), Tree('.', ['.'])]), ...]


 
 First NLTK and the corpus must be imported. To start, I'm using Penn Treebank for 
practice. Then I have to define the variable 'parsed_sents' as the parsed sentences 
from the corpus. Next I must get the productions from the parsed sentences of the 
corpus. First I make the variable productions an empty list. Then use a for loop to  iterate through each tree in the parsed sentences, remove unary nodes and ternary 
nodes in order to make the productions Chomsky Normal Form (CNF). 
tree.productions() finds the productions of each tree, and I use the for loop to 
collect the productions from each tree and add it to the productions list. The I 
print the length of the productions and the first one to make sure it works. (As 
you can see, if I print all the productions, it would be a very long list.



```python
productions = [] # define productions as a list
    
for tree in parsed_sents:
        # Turn the productions into Chomsky Normal Form
        tree.collapse_unary(collapsePOS = False) # Remove unary nodes except for leaves
        tree.chomsky_normal_form(horzMarkov = 2) # Remove ternary nodes
        productions += tree.productions()
        
print(len(productions))
print(productions[0])

```

    211968
    S -> NP-SBJ S|<VP-.>



Now I have to create a PCFG from those productions. To do that, I need a start 
symbol (S) and the productions. Then I'll use the nltk.induce_pcfg function to 
create the PCFG.



```python
# define start symbol
start_symbol = nltk.Nonterminal("S")
# induce a PCFG grammar from the list of prods 
pcfg = nltk.induce_pcfg(start_symbol, productions)

```


Next I will use the Viterber Parser. The Viterbi Parser is a bottom-up PCFG parser 
that uses dynamic programming to find the most likely parse for a sentence. It 
parses texts by filling in a "most likely constituent table," which records the 
most probable tree representation.
First I will name the parser, and use the pcfg as the grammar. Then I create the 
sentence I want it to parse into the variable 'sent.'



```python
vit = nltk.ViterbiParser(pcfg) #name the parser using the pcfg
sent1 = "It is me".split()      #create the first sentence to parse and split it
for tree1 in vit.parse(sent1):   #find the trees for that sentence
    print(tree1)
sent2 = "It is I".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S (NP-SBJ (PRP It)) (VP (VBZ is) (NP (PRP me)))) (p=1.17548e-09)
    (S (NP-SBJ (PRP It)) (VP (VBZ is) (NP (PRP I)))) (p=1.47589e-08)



According to the parser and the Penn Treebank PCFG, the probability of the sentence 
1 "It is me" is much lower than the probability for sentence 2 "It is I." This is a 
hugely important piece of information for my research. It shows that people would 
rather use the nominative case pronoun "I" in what should be the accusative case 
position, instead of the accusative case pronoun "me."
Next I am going to test different combinations of pronouns to find the most 
probable version of each pair.



```python
sent1 = "He and I".split()      #create the first sentence to parse and split it
for tree1 in vit.parse(sent1):   #find the trees for that sentence
    print(tree1)
sent2 = "He and me".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S (NP-SBJ (PRP He)) (VP (JJ and) (NP (PRP I)))) (p=4.04897e-13)
    (S (NP-SBJ (PRP He)) (VP (JJ and) (NP (PRP me)))) (p=3.22485e-14)



As you can see, this is a sentence fragment, so the tree node names are not exactly 
correct (it defines [and me] as a VP and [and] as an adjective, but we do see that 
the probability of "He and I" is higher than that of "He and me." Next I am going  to check if capitalization of 'he' will make a difference in the nodes' names and 
the probabilities. Basically I'm looking to see if the productions can make a 
sentence fragment correctly.



```python
sent1 = "he and I".split()      
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "he and me".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S (NP-SBJ (PRP he)) (VP (JJ and) (NP (PRP I)))) (p=1.2757e-12)
    (S (NP-SBJ (PRP he)) (VP (JJ and) (NP (PRP me)))) (p=1.01605e-13)



The probabilities are actually a tad higher than those of the pair with the 
capitalized 'he,' but the nodes are exactly the same. So I need to add a verb to 
the sentence to make the nodes fit into better places. 



```python
sent1 = "He and I go".split()      #added verb 'go'
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "He and me go".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ-54
        (PRP He)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP I)))
      (VP (VB go))) (p=8.37399e-12)
    (S
      (NP-SBJ-54
        (PRP He)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP me)))
      (VP (VB go))) (p=6.66955e-13)



Now the nodes are actually representing the parts of speech correctly. The 'He and 
I' sentence has a higher probability than 'he and me.'



```python
sent1 = "It is him and I".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "It is him and me".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ (PRP It))
      (VP
        (VBZ is)
        (NP (NP (PRP him)) (NP|<CC-NP> (CC and) (NP (PRP I)))))) (p=1.22494e-14)
    (S
      (NP-SBJ (PRP It))
      (VP
        (VBZ is)
        (NP (NP (PRP him)) (NP|<CC-NP> (CC and) (NP (PRP me)))))) (p=9.75619e-16)



The pronoun pair 'him and I' in accusative position is more probable than 'him and 
me' in accusative position according to the PCFG. Syntactically the more probable 
option is ungrammatical.



```python
sent1 = "It is she".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "It is her".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S (NP-SBJ (PRP It)) (VP (VBZ is) (NP (PRP she)))) (p=1.00569e-08)
    (S (NP-SBJ (PRP It)) (VP (VBZ is) (NP (PRP her)))) (p=2.35097e-09)



Again the more probable option when pronouns are in object position is ungrammatical by case.



```python
sent1 = "She and he go".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "She and him go".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ-54
        (PRP She)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP he)))
      (VP (VB go))) (p=4.90319e-12)
    (S
      (NP-SBJ-54
        (PRP She)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP him)))
      (VP (VB go))) (p=3.41091e-13)



Again the more probable option when pronouns are in subject position is the more grammatical option



```python
sent1 = "She and he go".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "He and she go".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ-54
        (PRP She)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP he)))
      (VP (VB go))) (p=4.90319e-12)
    (S
      (NP-SBJ-54
        (PRP He)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP she)))
      (VP (VB go))) (p=5.70617e-12)



```python
sent1 = "She and he were".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "He and she were".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ-54
        (PRP She)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP he)))
      (VP (VBD were))) (p=2.13936e-11)
    (S
      (NP-SBJ-54
        (PRP He)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP she)))
      (VP (VBD were))) (p=2.48972e-11)



I wanted to look at if speakers prefer to use 'he' before 'she' or vice versa. The two examples above show us that both pairs' probabilities are very close to each other. Also, speakers prefer to say 'she' before 'she' in subject position.



```python
sent1 = "It was her and him".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "It was him and her".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ (PRP It))
      (VP
        (VBD was)
        (NP
          (NP (PRP her))
          (NP|<CC-NP> (CC and) (NP (PRP him)))))) (p=1.0835e-15)
    (S
      (NP-SBJ (PRP It))
      (VP
        (VBD was)
        (NP
          (NP (PRP him))
          (NP|<CC-NP> (CC and) (NP (PRP her)))))) (p=1.0835e-15)



The probabilities of 'him and her' or 'her and him' in object position are exactly the same. Even when tried with a different verb.



```python
sent1 = "Her and me were".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "She and I were".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ
        (NP (PRP$ Her))
        (NP-SBJ|<CC-NP> (CC and) (NP (PRP me))))
      (VP (VBD were))) (p=3.56976e-18)
    (S
      (NP-SBJ-54
        (PRP She)
        (NP-SBJ-54|<CC-PRP> (CC and) (PRP I)))
      (VP (VBD were))) (p=1.05108e-11)



The example above is quite interesting because the first time I tried to set sent1 as 'Me and her were,' but got an error : the grammar did not cover 'Me' as an initial word in a sentence. This leads me to wonder if it only allowed 'Her' as an initial word because it is possessive. So I tried 'Him' as an intial word and got the same error.



```python
sent1 = "She gave it to him".split()      #change structure / pronouns
for tree1 in vit.parse(sent1):   
    print(tree1)
sent2 = "She gave it to he".split()      
for tree2 in vit.parse(sent2): 
    print(tree2)
```

    (S
      (NP-SBJ (PRP She))
      (VP
        (VBD gave)
        (VP|<NP-PP-CLR>
          (NP (PRP it))
          (PP-CLR (TO to) (NP (PRP him)))))) (p=8.01198e-16)
    (S
      (NP-SBJ (PRP She))
      (VP
        (VBD gave)
        (VP|<NP-PP-CLR>
          (NP (PRP it))
          (PP-CLR (TO to) (NP (PRP he)))))) (p=1.15172e-14)



The probabalistic parsing of these examples has shown that speakers care less about case in object position than in subject position. So far I noticed that when the pronoun pairs are in subject position, it is more probable that they will be the correct case. And when the pronoun pairs are in object position, it is more probable that they will not be in the correct case. I'd like to continue this research with other parsed corpuses.

