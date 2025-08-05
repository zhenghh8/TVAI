'''
Copied from: https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

Modified by: Maxlinn

1. adapt calculation of CHAIR-i and CHAIR-s for Python3, supports for both json and jsonl file input.
2. integrate synonyms.txt to make the script standalone.
3. remove machine-translation based metrics BLEU-n, CIDEr, ROGUE
4. add new metric Recall, which represents the node words(i.e. lemmas of objects) coverage overall.
5. add pickle cache mechanism to make it fast for repetitive evaluations.
'''


import os
import sys
import nltk
import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import argparse
import tqdm
import pickle
from collections import defaultdict
import numpy as np
from visualization import visualize, attn_sorted

#### for attention statistics
MODEL = 'llava-1.5'  # 'llava-1.5' or 'qwen-vl-chat'
#### for attention statistics

# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def combine_coco_captions(annotation_path):

    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps 

def combine_coco_instances(annotation_path):

    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances 

class CHAIR(object):

    def __init__(self, coco_path):

        self.imid_to_objects = defaultdict(list) # later become a dict of sets

        self.coco_path = coco_path

        # read in synonyms
        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO
        
        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']
        
        # double_word_dict will map double words to the word they should be treated as in our analysis
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):

        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        # Read in captions 
        #### for attention statistics        
        self.caps, self.eval_imids, self.attns_img, self.attns_text, self.words_ids = load_generated_captions(cap_file, image_id_key, caption_key)
        #### for attention statistics  

        # self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):
    
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''
        #### for attention statistic
        # standard preprocessing
        one_token_word = ' newline '

        if MODEL == 'llava-1.5':
            special_one_llava = ['\n', '."', '</s>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',"']
            for special in special_one_llava:
                caption = caption.replace(special, one_token_word)
        # qwen-vl-chat
        elif MODEL == 'qwen-vl-chat':
            special_one_qwen = ['."\n\n', '.\n\n', '\n\n', '.\n', '\n', '<|im_start|>', '<|im_end|>', '<|endoftext|>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '."', ',"', '".', 'cannot']
            for special in special_one_qwen:
                caption = caption.replace(special, one_token_word)
        else:
            # TODO
            raise NotImplementedError
        #### for attention statistics
            
        words_ = nltk.word_tokenize(caption.lower())

        #### for attention statistics
        words = [word.replace('newline', '\n') for word in words_]
        #### for attention statistic
        # words = words_
    
        # lemmatize
        tagged_sent = nltk.pos_tag(words)  # part of speech
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        # words = [singularize(w) for w in words]
        words = lemmas_sent
    
        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        idxs.append(i)
        words = double_words
    
        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): 
            words = [word for word in words if word != 'seat']
            idxs = [idx for idx, word in zip(idxs, words) if word != 'seat']
    
        # get synonyms for all words in the caption
        idxs_object = [idx for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        #return all the MSCOCO objects in the caption
        return words, node_words, idxs, idxs_object, double_words

    def get_annotations_from_segments(self):
        '''
        Add objects taken from MSCOCO segmentation masks
        '''

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        # make dict linking object name to ids
        id_to_name = {} #dict with id to synsets 
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" 
                              %(i, len(segment_annotations)))
            imid = annotation['image_id']
            
            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        '''
        Add objects taken from MSCOCO ground truth captions 
        '''

        coco_caps = combine_coco_captions(self.coco_path)

        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions' 
                              %(i, len(coco_caps['annotations'])))
            imid = annotation['image_id']
            
            # get words in mscoco_objects from caption (through synonyms convertion)
            _, node_words, _, _ = self.caption_to_words(annotation['caption'])  
            # note here is update, so call get_annotations_from_segments first
            self.imid_to_objects[imid].extend(node_words)
        print("\n")


    def get_annotations(self):

        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''
        
        self.get_annotations_from_segments() 
        self.get_annotations_from_captions()
        # deduplicate (annotations from segments and captions may overlap)
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, ana_file, image_id_key, caption_key):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''
        self._load_generated_captions_into_evaluator(os.path.join('results_generation', ana_file), image_id_key, caption_key)
        caps = self.caps
        eval_imids = self.eval_imids 

        #### for attention statistics
        attns_img = self.attns_img
        attns_text = self.attns_text
        words_ids = self.words_ids
        #### for attention statistics
        
        imid_to_objects = self.imid_to_objects
        
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        len_caps = 0.
        
        # :add:
        num_recall_gt_objects = 0.
        num_gt_objects = 0.
        num_generated_objects = 0.

        output = {'sentences': []} 

        #### for attention statistics
        delimiter = ' '
        hall_flag = 0
        nonhall_flag =0
        save_attn_label = os.path.join('results_analysis', ana_file + '_attn&label.txt')
        count_inspect = 0
        #### for attention statistics

        #### for attention statistics
        # save visualization of each instance
        if not os.path.exists('results_analysis'):
            os.makedirs('results_analysis')
        if not os.path.exists('results_analysis/chair'):
            os.makedirs('results_analysis/chair')
        base_name = os.path.splitext(os.path.basename(args.ans_file))[0]
        dir_name = os.path.dirname(args.ans_file)
        if not os.path.exists(os.path.join('results_analysis', dir_name)):
            os.makedirs(os.path.join('results_analysis', dir_name))
        attn_save_dir = os.path.join('results_analysis', dir_name, base_name)
        if not os.path.exists(attn_save_dir):
            os.makedirs(attn_save_dir)
        if not os.path.exists(attn_save_dir + '/no_hall_words'):
            os.makedirs(attn_save_dir + '/no_hall_words')
        if not os.path.exists(attn_save_dir + '/hall_words'):
            os.makedirs(attn_save_dir + '/hall_words')
        #### for attention statistics
        
        # with open(save_attn_label, 'w', encoding='utf-8') as file: 
        for i in tqdm.trange(len(caps)):
            cap :str = caps[i]
            imid :int = eval_imids[i]
    
            # get all words in the caption, as well as corresponding node word
            # pos = cap.rfind('.')
            # cap = cap[:pos+1]
            words, node_words, idxs, idxs_object, raw_words = self.caption_to_words(cap) 

            gt_objects = imid_to_objects[imid]
            cap_dict = {'image_id': imid, 
                        'caption': cap,
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words), 
                        'mscoco_hallucinated_words': [], 
                        'hallucination_idxs': [], 
                        'mscoco_non_hallucinated_words': [], 
                        'non_hallucination_idxs': [], 
                        'words': raw_words 
                        }

            # :add:
            cap_dict['metrics'] = {'CHAIRs': 0,
                                'CHAIRi': 0,
                                'Recall': 0,
                                'Len': 0,
                                }

            # count hallucinated words
            coco_word_count += len(node_words)
            hallucinated = False
            
            
            # obtain the attention statistics
            recall_gt_objects = set()
            for idx, (word, node_word) in enumerate(zip(words, node_words)):
                #### for attention statistics
                # print(cap)
                # print(word)
                # print(words_ids[i][idxs[idxs_object[idx]]], words_ids[i][idxs[idxs_object[idx]+1]])
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append((idxs[idxs_object[idx]], idxs[idxs_object[idx]+1]))
                    hallucinated = True

                    # attn_img_first = np.array(attns_img[i][words_ids[i][idxs[idxs_object[idx]]]])
                    attn_img_mean = np.mean(attns_img[i][words_ids[i][idxs[idxs_object[idx]]]:words_ids[i][idxs[idxs_object[idx]+1]]], axis=0)
                    # attn_text_first = np.array(attns_text[i][words_ids[i][idxs[idxs_object[idx]]]])
                    attn_text_mean = np.mean(attns_text[i][words_ids[i][idxs[idxs_object[idx]]]:words_ids[i][idxs[idxs_object[idx]+1]]], axis=0)

                    # # save_dir = os.path.join(attn_save_dir, 'hall_words', 'img_first_{}_'.format(words) + str(imid) + '.png')
                    # # visualize(attn_img_first, 'hall_img_first', cmap='viridis', save_dir=save_dir)
                    save_dir = os.path.join(attn_save_dir, 'hall_words', 'img_mean_{}_'.format(word) + str(imid) + '.png')
                    visualize(attn_img_mean, 'VAR', cmap='viridis', save_dir=save_dir)

                    # # save_dir = os.path.join(attn_save_dir, 'hall_words', 'text_first_{}_'.format(words) + str(imid) + '.png')
                    # # visualize(attn_text_first, 'hall_text_first', cmap='viridis', save_dir=save_dir)
                    save_dir = os.path.join(attn_save_dir, 'hall_words', 'text_mean_{}_'.format(word) + str(imid) + '.png')
                    visualize(attn_text_mean, 'TAR', cmap='viridis', save_dir=save_dir)

                    # '''
                    # attn_img_first_ = np.reshape(attn_img_first, -1).astype(str)
                    # attn_img_mean_ = np.reshape(attn_img_mean, -1).astype(str)
                    # attn_text_first_ = np.reshape(attn_text_first, -1).astype(str)
                    # attn_text_mean_ = np.reshape(attn_text_mean, -1).astype(str)

                    # file.write('image_id: ' + str(imid) + ' objedct: ' + str(node_word) + ' ' + delimiter.join(attn_img_mean_) + \
                    #             ' ' + delimiter.join(attn_text_mean_) + ' ' + delimiter.join(attn_img_first_) + ' ' \
                    #             + delimiter.join(attn_text_first_) + ' ' + '1' + '\n')
                    count_inspect += 1 

                    
                    if hall_flag == 0:
                        mean_hall_img_attn_mean = attn_img_mean
                        mean_hall_text_attn_mean = attn_text_mean
                        # mean_hall_img_attn_first = attn_img_first
                        # mean_hall_text_attn_first = attn_text_first
                        hall_flag = 1
                    else:
                        mean_hall_img_attn_mean = 0.5 * (mean_hall_img_attn_mean + attn_img_mean)
                        mean_hall_text_attn_mean = 0.5 * (mean_hall_text_attn_mean + attn_text_mean)
                        # mean_hall_img_attn_first = 0.5 * (mean_hall_img_attn_first + attn_img_first)
                        # mean_hall_text_attn_first = 0.5 * (mean_hall_text_attn_first + attn_text_first)
                    # '''

                else:
                    cap_dict['mscoco_non_hallucinated_words'].append((word, node_word))
                    cap_dict['non_hallucination_idxs'].append((idxs[idxs_object[idx]], idxs[idxs_object[idx]+1]))
                    recall_gt_objects.add(node_word)

                    # attn_img_first = np.array(attns_img[i][words_ids[i][idxs[idxs_object[idx]]]])
                    attn_img_mean = np.mean(attns_img[i][words_ids[i][idxs[idxs_object[idx]]]:words_ids[i][idxs[idxs_object[idx]+1]]], axis=0)
                    # attn_text_first = np.array(attns_text[i][words_ids[i][idxs[idxs_object[idx]]]])
                    attn_text_mean = np.mean(attns_text[i][words_ids[i][idxs[idxs_object[idx]]]:words_ids[i][idxs[idxs_object[idx]+1]]], axis=0)

                    # save_dir = os.path.join(attn_save_dir, 'no_hall_words', 'img_first_{}_'.format(word) + str(imid) + '.png')
                    # visualize(attn_img_first, 'no_hall_img_first', cmap='viridis', save_dir=save_dir)
                    save_dir = os.path.join(attn_save_dir, 'no_hall_words', 'img_mean_{}_'.format(word) + str(imid) + '.png')
                    visualize(attn_img_mean, 'VAR', cmap='viridis', save_dir=save_dir)

                    # save_dir = os.path.join(attn_save_dir, 'no_hall_words', 'text_first_{}_'.format(word) + str(imid) + '.png')
                    # visualize(attn_text_first, 'no_hall_text_first', cmap='viridis', save_dir=save_dir)
                    save_dir = os.path.join(attn_save_dir, 'no_hall_words', 'text_mean_{}_'.format(word) + str(imid) + '.png')
                    visualize(attn_text_mean, 'TAR', cmap='viridis', save_dir=save_dir)

                    # '''
                    # attn_img_first_ = np.reshape(attn_img_first, -1).astype(str)
                    # attn_img_mean_ = np.reshape(attn_img_mean, -1).astype(str)
                    # attn_text_first_ = np.reshape(attn_text_first, -1).astype(str)
                    # attn_text_mean_ = np.reshape(attn_text_mean, -1).astype(str)

                    # file.write('image_id: ' + str(imid) + ' objedct: ' + str(node_word) + ' ' + delimiter.join(attn_img_mean_) + \
                    #             ' ' + delimiter.join(attn_text_mean_) + ' ' + delimiter.join(attn_img_first_) + ' ' \
                    #             + delimiter.join(attn_text_first_) + ' ' + '0' + '\n')
                    # count_inspect += 1 

                    if nonhall_flag == 0:
                        mean_nonhall_img_attn_mean = attn_img_mean
                        mean_nonhall_text_attn_mean = attn_text_mean
                        # mean_nonhall_img_attn_first = attn_img_first
                        # mean_nonhall_text_attn_first = attn_text_first
                        nonhall_flag =1
                    else:
                        mean_nonhall_img_attn_mean = 0.5 * (mean_nonhall_img_attn_mean + attn_img_mean)
                        mean_nonhall_text_attn_mean = 0.5 * (mean_nonhall_text_attn_mean + attn_text_mean)
                        # mean_nonhall_img_attn_first = 0.5 * (mean_nonhall_img_attn_first + attn_img_first)
                        # mean_nonhall_text_attn_first = 0.5 * (mean_nonhall_text_attn_first + attn_text_first)
                    # '''
                #### for attention statistics
    
            #count hallucinated caps
            num_caps += 1
            len_caps += len(raw_words)
            if hallucinated:
                num_hallucinated_caps += 1
            
            # add
            num_gt_objects += len(gt_objects)
            num_generated_objects += len(set(node_words))  # deduplication
            num_recall_gt_objects += len(recall_gt_objects)
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            cap_dict['metrics']['Recall'] = 0.
            cap_dict['metrics']['Len'] = 0.
    
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))
            
            # add
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)
                if len(node_words) == 0:
                    cap_dict['metrics']['Precision'] = 0.
                else:
                    cap_dict['metrics']['Precision'] = len(recall_gt_objects) / len(set(node_words))
                if (cap_dict['metrics']['Precision'] + cap_dict['metrics']['Recall']) == 0:
                    cap_dict['metrics']['F1'] = 0.
                else:
                    cap_dict['metrics']['F1'] = 2 * (cap_dict['metrics']['Recall'] * cap_dict['metrics']['Precision']) / (cap_dict['metrics']['Precision'] + cap_dict['metrics']['Recall'])

            output['sentences'].append(cap_dict)
 
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        # add
        recall = num_recall_gt_objects / num_gt_objects
        precision = num_recall_gt_objects / num_generated_objects
        f1 = 2 * (recall * precision) / (precision + recall)
        avg_len = (0.01*len_caps/num_caps)
    
        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'Recall': recall,
                                     'Precision': precision,
                                     'F1': f1,
                                     'Len': avg_len,}
        
        # assert count_inspect == coco_word_count, \
        # "the number of attention_hallucination records is not equal to the number of generated objects (before deduplication)!"
        
        output['base_information'] = {
            'num_generated_objects (before deduplication)': coco_word_count,
            'num_generated_objects (after deduplication)': num_generated_objects, 
        }
        
        save_detail = os.path.join('results_analysis', ana_file + '_detail.json')
        save_hallucinated_words(save_detail, output)


        #### for attention statistics
        # # visualization

        # # sorted
        # # mean
        # mean_hall_img_attn_mean = attn_sorted(mean_hall_img_attn_mean, dim=1)
        # mean_hall_text_attn_mean = attn_sorted(mean_hall_text_attn_mean, dim=1)

        # mean_nonhall_img_attn_mean = attn_sorted(mean_nonhall_img_attn_mean, dim=1)
        # mean_nonhall_text_attn_mean = attn_sorted(mean_nonhall_text_attn_mean, dim=1)

        # # first 
        # mean_hall_img_attn_first = attn_sorted(mean_hall_img_attn_first, dim=1)
        # mean_hall_text_attn_first = attn_sorted(mean_hall_text_attn_first, dim=1)

        # mean_nonhall_img_attn_first = attn_sorted(mean_nonhall_img_attn_first, dim=1)
        # mean_nonhall_text_attn_first = attn_sorted(mean_nonhall_text_attn_first, dim=1)

        visualize(mean_hall_img_attn_mean, 'VAR', save_dir=os.path.join('results_analysis', ana_file + '_hall_img_mean.png'), vmax=0.7)
        visualize(mean_nonhall_img_attn_mean, 'VAR')
        visualize(mean_hall_text_attn_mean, 'TAR', save_dir=os.path.join('results_analysis', ana_file + '_hall_txt_mean.png'), vmax=0.7)
        visualize(mean_nonhall_text_attn_mean, 'TAR')
        # visualize(mean_hall_img_attn_first, 'hall_img_first')
        # visualize(mean_nonhall_img_attn_first, 'nonhall_img_first')
        # visualize(mean_hall_text_attn_first, 'hall_text_first')
        # visualize(mean_nonhall_text_attn_first, 'nonhall_text_first')

        np.save(os.path.join('results_analysis', ana_file + '_hall_img_mean.npy'), mean_hall_img_attn_mean)
        np.save(os.path.join('results_analysis', ana_file + '_hall_text_mean.npy'), mean_hall_text_attn_mean)
        np.save(os.path.join('results_analysis', ana_file + '_nonhall_img_mean.npy'), mean_nonhall_img_attn_mean)
        np.save(os.path.join('results_analysis', ana_file + '_nonhall_text_mean.npy'), mean_nonhall_text_attn_mean)

        # plt.show()
        ### for attention statistics

        return output 

def load_generated_captions(cap_file, image_id_key:str, caption_key:str):
    # Read in captions        
    # it should be list of dict
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        result_dict = json.load(open(cap_file))
    elif ext == '.jsonl':
        result_dict = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f'Unspported extension {ext} for cap_file: {cap_file}')

    # list of int
    imids = [obj[image_id_key] for obj in result_dict]
    
    # list of str
    caps = [obj[caption_key] for obj in result_dict]

    ### for attention statistics
    attn_img_key = 'generated_attn_img'
    attn_text_key = 'generated_attn_text'
    word_id_key = 'word_ids'

    
    # list of float
    attns_img = [obj[attn_img_key] for obj in result_dict]
    attns_text = [obj[attn_text_key] for obj in result_dict]

    # list of int
    words_ids = [obj[word_id_key] for obj in result_dict]
    ### for attention statistics
       
    return caps, imids, attns_img, attns_text, words_ids
    # return caps, imids

def save_hallucinated_words(cap_file, cap_dict): 
    with open(cap_file, 'a') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    
    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(10)
        v_str = f'{v * 100:.01f}'
        print(k_str, v_str, sep=': ')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ans_file", type=str, default='',
                        help="path towards json or jsonl saving image ids and their captions in list of dict.")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="in each dict of cap_file, which key stores image id of coco.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="in each dict of cap_file, which key stores caption of the image.")
    
    parser.add_argument("--coco_path", type=str, default='coco/annotations_trainval2014/annotations',
                        help="only use for regenerating CHAIR evaluator object, will be ignored if uses cached evaluator.")

    
    args = parser.parse_args()

    cache_path = 'chair.pkl'
    if os.path.exists(cache_path):
        evaluator = pickle.load(open(cache_path, 'rb'))
        print(f"loaded evaluator from cache: {cache_path}")
    else:
        print(f"cache not setted or not exist yet, building from scratch...")
        evaluator = CHAIR(args.coco_path)
        pickle.dump(evaluator, open(cache_path, 'wb'))
        print(f"cached evaluator to: {cache_path}")

    cap_dict = evaluator.compute_chair(args.ans_file, args.image_id_key, args.caption_key) 
    print_metrics(cap_dict)
    