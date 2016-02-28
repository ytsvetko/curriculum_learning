import sys
import numpy as np
import optparse
import logging
from collections import OrderedDict
import os
import time 

# Import library paths
sys.path.append('/usr1/home/ytsvetko/projects/curric/downstream/ner/scripts')
sys.path.append('/usr1/home/ytsvetko/projects/curric/downstream/ner/scripts')

# Load UltraDeep / Tools / Networks library
from UltraDeep.experiment import Experiment
from UltraDeep.utils import get_experiment_name  # , parse_experiment_name
import tools.datasets as datasets
from tools.utils.loader import pad_sentence_tokens_chars, add_caps_features
from tools.utils.loader import convert_tagging_scheme
from tools.utils.evaluate import evaluate_pos, evaluate_conll, evaluate_genia
import networks
from tools.utils.loader import get_singletons, insert_singletons
from tools.utils.loader import replace_sentences_digits_with_0
from tools.utils.loader import augment_with_pretrained

#
# Initialize experiment
#
parameters = OrderedDict()
if True:
    # Load parameters from command line
    optparser = optparse.OptionParser()
    optparser.add_option("-a", "--data_type", dest="data_type", default="pos", help="Data type")
    optparser.add_option("-b", "--tag_scheme", dest="tag_scheme", default="iobes", help="Tagging scheme (IOB or IOBES)")
    optparser.add_option("-c", "--char_emb_dim", dest="char_emb_dim", default="30", help="Char embedding dimension")
    optparser.add_option("-d", "--char_lstm_dim", dest="char_lstm_dim", default="50", help="Char LSTM hidden layer size")
    optparser.add_option("-e", "--char_bidirect", dest="char_bidirect", default="0", help="Bidirectional LSTM for chars")
    optparser.add_option("-f", "--token_emb_dim", dest="token_emb_dim", default="50", help="Token embedding dimension")
    optparser.add_option("-g", "--token_lstm_dim", dest="token_lstm_dim", default="50", help="Token LSTM hidden layer size")
    optparser.add_option("-i", "--token_bidirect", dest="token_bidirect", default="1", help="Bidirectional LSTM for tokens")
    optparser.add_option("-j", "--dropout", dest="dropout", default="0", help="Droupout on the input")
    optparser.add_option("-k", "--lr_method", dest="lr_method", default="adam", help="Learning method")
    optparser.add_option("-l", "--ext_emb", dest="ext_emb", default="", help="Location of external embeddings")
    optparser.add_option("-m", "--crf", dest="crf", default="0", help="Use CRF")
    optparser.add_option("-n", "--mask", dest="mask", default="0", help="Use a mask for CRF")
    optparser.add_option("-o", "--dynamic", dest="dynamic", default="0", help="Dynamic CRF")
    optparser.add_option("-p", "--caps_dim", dest="caps_dim", default="0", help="Include Capitalization features")
    optparser.add_option("-q", "--lower_t", dest="lower_t", default="0", help="Lowercase tokens")
    optparser.add_option("-r", "--lower_c", dest="lower_c", default="0", help="Lowercase chars")
    optparser.add_option("-s", "--gaz_dim", dest="gaz_dim", default="0", help="Gazetteers dimension")
    # optparser.add_option("-t", "--char_model_path", dest="char_model_path", default="", help="Pretrained character level model path")
    # optparser.add_option("-u", "--p_token", dest="p_token", default="", help="Pretrain token embeddings from char model")
    # optparser.add_option("-v", "--p_char", dest="p_char", default="", help="Pretrain char embedings from char model")
    optparser.add_option("-w", "--zeros", dest="zeros", default="1", help="Replace digits with 0")
    optparser.add_option("-x", "--pos_dim", dest="pos_dim", default="0", help="Use POS-tags")
    optparser.add_option("-y", "--comment", dest="comment", default="", help="Comment on experiment")
    (opts, args) = optparser.parse_args()
    # Parse arguments
    parameters['data_type'] = opts.data_type
    parameters['tag_scheme'] = opts.tag_scheme
    parameters['char_emb_dim'] = int(opts.char_emb_dim)
    parameters['char_lstm_dim'] = int(opts.char_lstm_dim)
    parameters['char_bidirect'] = opts.char_bidirect == '1'
    parameters['token_emb_dim'] = int(opts.token_emb_dim)
    parameters['token_lstm_dim'] = int(opts.token_lstm_dim)
    parameters['token_bidirect'] = opts.token_bidirect == '1'
    parameters['dropout'] = opts.dropout == '1'
    parameters['lr_method'] = str(opts.lr_method)
    parameters['ext_emb'] = opts.ext_emb
    parameters['crf'] = opts.crf == '1'
    parameters['mask'] = opts.mask == '1'
    parameters['caps_dim'] = int(opts.caps_dim)
    parameters['lower_t'] = int(opts.lower_t)
    parameters['lower_c'] = int(opts.lower_c)
    parameters['gaz_dim'] = int(opts.gaz_dim)
    # parameters['char_model_path'] = opts.char_model_path
    # parameters['p_token'] = opts.p_token == '1'
    # parameters['p_char'] = opts.p_char == '1'
    parameters['zeros'] = opts.zeros == '1'
    parameters['pos_dim'] = int(opts.pos_dim)
    # parameters['dynamic'] = opts.dynamic == '1'
    # assert not (parameters['dynamic'] and not parameters['crf'])
    if opts.comment:
        parameters['comment'] = opts.comment
else:
    # Use default parameters
    parameters['data_type'] = 'genia'
    parameters['tag_scheme'] = 'iobes'
    parameters['char_emb_dim'] = 31
    parameters['char_lstm_dim'] = 50
    parameters['char_bidirect'] = '1' == '1'
    parameters['token_emb_dim'] = 52
    parameters['token_lstm_dim'] = 52
    parameters['token_bidirect'] = '1' == '1'
    parameters['dropout'] = False
    parameters['lr_method'] = 'adam'
    parameters['ext_emb'] = ''
    parameters['crf'] = False
    parameters['mask'] = False
    parameters['caps_dim'] = 0
    parameters['lower_t'] = False
    parameters['lower_c'] = False
    parameters['gaz_dim'] = 0
    # parameters['char_model_path'] = ''
    # parameters['p_token'] = False
    # parameters['p_char'] = False
    parameters['zeros'] = True
    parameters['pos_dim'] = 0
    # parameters['dynamic'] = False
    # assert not (parameters['dynamic'] and not parameters['crf'])
    if False:
        parameters['comment'] = ''

# parameters['comment'] = 'fixV'

# assert not (parameters['p_token'] or parameters['p_char'])
assert (parameters['data_type'] == 'pos' and parameters['tag_scheme'] == '') or parameters['tag_scheme'] in ['iob', 'iobes']
# assert parameters['ext_emb'] == '' or not parameters['p_token']
# assert (parameters['char_model_path'] == '' and (not parameters['p_token']) and (not parameters['p_char'])) or (os.path.isdir(parameters['char_model_path']) and (parameters['p_token'] or parameters['p_char']))
assert parameters['token_emb_dim'] > 0 or parameters['ext_emb'] == '' # and not parameters['p_token']
assert parameters['pos_dim'] == 0 or parameters['data_type'] in ['ner', 'german', 'spanish', 'chunk', 'dutch']

# Character-based model for pretraining
# if parameters['char_model_path']:
#     char_parameters = parse_experiment_name(os.path.basename(parameters['char_model_path']))
#     assert char_parameters['data_type'] == parameters['data_type']
#     assert char_parameters['caps_dim'] == 0 and not char_parameters['dropout']
#     # If we pre-initialize characters, we need to have trained the characters
#     # using identical parameters
#     if parameters['p_char']:
#         assert all(char_parameters[x] == parameters[x] for x in ['char_emb_dim', 'char_lstm_dim', 'char_bidirect', 'lower_c'])
#     # If we want to pre-initialize tokens, we need to have tokens
#     if parameters['p_token']:
#         assert parameters['token_emb_dim'] == char_parameters['char_lstm_dim'] * (1 + char_parameters['char_bidirect']) + char_parameters['caps_dim']

# Initialize experiment
timestr = time.strftime("%Y%m%d-%H-%M-")
experiment_name = parameters['data_type']+"-"+os.path.basename(os.path.dirname(parameters['ext_emb']))+"-"+os.path.basename(parameters['ext_emb']) #get_experiment_name(OrderedDict([(k, v) for k, v in parameters.items() if k not in ['char_model_path', 'tag_scheme', 'mask', 'p_char', 'p_token']]))  # , 'zeros'
experiment_name = experiment_name.replace('comment=', 'com=')
experiment = Experiment(
    name=experiment_name,
    dump_path='/usr1/home/ytsvetko/projects/curric/downstream/ner/tmp/'
)
print "Experiment location: %s" % experiment.dump_path
logger = logging.getLogger()

#
# Load data
#
if parameters['data_type'] == 'pos':
    # Load data
    train_sentences, dev_sentences, test_sentences = datasets.penn_tb.loader.load_sentences(True)

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.penn_tb.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.penn_tb.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.penn_tb.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.penn_tb.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_tags, tag_to_index, index_to_tag = datasets.penn_tb.loader.get_postags_dico_mappings(train_sentences, dev_sentences, test_sentences)

    # Prepare data
    train_data = datasets.penn_tb.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.penn_tb.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.penn_tb.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda _, parsed_sentences, A_t: evaluate_pos(
        parameters, f_eval, parsed_sentences, index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'ner':
    # Load data
    train_sentences, dev_sentences, test_sentences = datasets.english_ner.loader.load_sentences()

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.english_ner.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.english_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.english_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.english_ner.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_postags, postag_to_index, index_to_postag = datasets.english_ner.loader.get_postags_dico_mappings(train_sentences)
    dictionary_chunktags, chunktag_to_index, index_to_chunktag = datasets.english_ner.loader.get_chunktags_dico_mappings(train_sentences)
    dictionary_tags, tag_to_index, index_to_tag = datasets.english_ner.loader.get_netags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.english_ner.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, postag_to_index, chunktag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.english_ner.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, postag_to_index, chunktag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.english_ner.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, postag_to_index, chunktag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Gazetteers
    if parameters['gaz_dim']:
        gazetteers_train = datasets.english_ner.loader.extract_gazetteers(train_sentences)
        gazetteers_dataset = datasets.english_ner.loader.load_gazetteers()
        # token_to_gazetteers = datasets.english_ner.loader.load_token_to_gazetteers([gazetteers_train, gazetteers_dataset])
        # datasets.english_ner.loader.add_gazetteers(train_data, token_to_gazetteers, index_to_token)
        # datasets.english_ner.loader.add_gazetteers(dev_data, token_to_gazetteers, index_to_token)
        # datasets.english_ner.loader.add_gazetteers(test_data, token_to_gazetteers, index_to_token)
        token_to_chunks = datasets.english_ner.loader.load_token_to_gazetteers_chunks([gazetteers_train, gazetteers_dataset])
        datasets.english_ner.loader.add_gazetteers(train_data, token_to_chunks, index_to_token)
        datasets.english_ner.loader.add_gazetteers(dev_data, token_to_chunks, index_to_token)
        datasets.english_ner.loader.add_gazetteers(test_data, token_to_chunks, index_to_token)

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_conll(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'chunk':
    # Load data
    train_dev_sentences, test_sentences = datasets.chunking.loader.load_sentences()
    train_sentences = train_dev_sentences[:-1000]
    dev_sentences = train_dev_sentences[-1000:]

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.chunking.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.chunking.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.chunking.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.chunking.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_postags, postag_to_index, index_to_postag = datasets.chunking.loader.get_postags_dico_mappings(train_sentences)
    dictionary_tags, tag_to_index, index_to_tag = datasets.chunking.loader.get_chunktags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.chunking.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.chunking.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.chunking.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_conll(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'genia':
    # Load data
    train_dev_sentences, test_sentences = datasets.genia.loader.load_sentences()
    delim = int(0.2 * len(train_dev_sentences))
    train_sentences = train_dev_sentences[delim:]
    dev_sentences = train_dev_sentences[:delim]

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.genia.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.genia.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.genia.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.genia.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_tags, tag_to_index, index_to_tag = datasets.genia.loader.get_netags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.genia.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.genia.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.genia.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_genia(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'genia_pos':
    # Load data
    train_test_dev_sentences = datasets.genia_pos.loader.load_sentences()
    n_sentences = len(train_test_dev_sentences)
    train_sentences = train_test_dev_sentences[:int(.8 * n_sentences)]
    dev_sentences = train_test_dev_sentences[int(.8 * n_sentences):int(.9 * n_sentences)]
    test_sentences = train_test_dev_sentences[int(.9 * n_sentences):]

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.genia_pos.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    dictionary_tokens, token_to_index, index_to_token = datasets.genia_pos.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.genia_pos.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_tags, tag_to_index, index_to_tag = datasets.genia_pos.loader.get_postags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.genia_pos.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'], first_tag=False)
    dev_data = datasets.genia_pos.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'], first_tag=True)
    test_data = datasets.genia_pos.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'], first_tag=True)
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda _, parsed_sentences, A_t: evaluate_pos(
        parameters, f_eval, parsed_sentences, index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'german':
    # Load data
    train_sentences, dev_sentences, test_sentences = datasets.german_ner.loader.load_sentences()

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.german_ner.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.german_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.german_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.german_ner.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_postags, postag_to_index, index_to_postag = datasets.german_ner.loader.get_postags_dico_mappings(train_sentences)
    dictionary_tags, tag_to_index, index_to_tag = datasets.german_ner.loader.get_netags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.german_ner.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.german_ner.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.german_ner.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_conll(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'chinese':
    # Load data
    train_sentences, dev_sentences, test_sentences = datasets.chinese_ner.loader.load_sentences()

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.chinese_ner.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.chinese_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.chinese_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.chinese_ner.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_tags, tag_to_index, index_to_tag = datasets.chinese_ner.loader.get_netags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.chinese_ner.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.chinese_ner.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.chinese_ner.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_conll(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'dutch':
    # Load data
    train_sentences, dev_sentences, test_sentences = datasets.dutch_ner.loader.load_sentences()

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.dutch_ner.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.dutch_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.dutch_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.dutch_ner.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_postags, postag_to_index, index_to_postag = datasets.dutch_ner.loader.get_postags_dico_mappings(train_sentences)
    dictionary_tags, tag_to_index, index_to_tag = datasets.dutch_ner.loader.get_netags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.dutch_ner.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.dutch_ner.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.dutch_ner.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_conll(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

elif parameters['data_type'] == 'spanish':
    # Load data
    train_sentences, dev_sentences, test_sentences = datasets.spanish_ner.loader.load_sentences()

    # zeros
    if parameters['zeros']:
        replace_sentences_digits_with_0(train_sentences)
        replace_sentences_digits_with_0(dev_sentences)
        replace_sentences_digits_with_0(test_sentences)

    # Tagging scheme
    if parameters['tag_scheme'] == 'iobes':
        convert_tagging_scheme(train_sentences, 'iob', 'iobes')
        convert_tagging_scheme(dev_sentences, 'iob', 'iobes')
        convert_tagging_scheme(test_sentences, 'iob', 'iobes')

    # Create dictionaries and mappings
    # if parameters['p_token']:
    #     dictionary_tokens, token_to_index, index_to_token = datasets.spanish_ner.loader.get_tokens_dico_mappings(train_sentences + dev_sentences + test_sentences, parameters['lower_t'])
    # else:
    if parameters['ext_emb']:
        dictionary_tokens_train, _, _ = datasets.spanish_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
        dictionary_tokens, token_to_index, index_to_token = augment_with_pretrained({k: v for k, v in dictionary_tokens_train.items()}, parameters['ext_emb'], dev_sentences + test_sentences)
    else:
        dictionary_tokens, token_to_index, index_to_token = datasets.spanish_ner.loader.get_tokens_dico_mappings(train_sentences, parameters['lower_t'])
    dictionary_chars, char_to_index, index_to_char = datasets.spanish_ner.loader.get_chars_dico_mappings(train_sentences, parameters['lower_c'])
    dictionary_postags, postag_to_index, index_to_postag = datasets.spanish_ner.loader.get_postags_dico_mappings(train_sentences)
    dictionary_tags, tag_to_index, index_to_tag = datasets.spanish_ner.loader.get_netags_dico_mappings(train_sentences)

    # Prepare data
    train_data = datasets.spanish_ner.loader.prepare_dataset(train_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    dev_data = datasets.spanish_ner.loader.prepare_dataset(dev_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    test_data = datasets.spanish_ner.loader.prepare_dataset(test_sentences, token_to_index, char_to_index, postag_to_index, tag_to_index, lower_tokens=parameters['lower_t'], lower_chars=parameters['lower_c'])
    logger.info("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # Evaluation function
    evaluate = lambda raw_sentences, parsed_sentences, A_t: evaluate_conll(
        parameters, f_eval, raw_sentences, parsed_sentences,
        index_to_tag, dictionary_tags,
        transition_scores=(A_t.get_value() if A_t is not None else None)
    )

else:
    raise Exception('Unknown data type!')


# Add caps features
if parameters['caps_dim']:
    logger.info('Adding caps features...')
    add_caps_features(train_data)
    add_caps_features(dev_data)
    add_caps_features(test_data)

print index_to_tag

#
# Build network
#
f_train, f_eval, A_t = networks.sequence_tagging_char_tok_lstm_crf.build(
    experiment, parameters, index_to_token, index_to_char, index_to_tag
)


#
# Train network
#
singletons = get_singletons(
    dictionary_tokens_train if parameters['ext_emb'] else dictionary_tokens,
    token_to_index
)
experiment.reset_time()
nb_epochs = 3
best_dev_score = -np.inf
best_test_score = -np.inf
count_total = 0
for epoch in xrange(nb_epochs):
    epoch_costs = []
    last_costs = []
    logger.info("Starting epoch %i..." % epoch)
    for i, index in enumerate(np.random.permutation(len(train_data))):
    # for i, index in enumerate(xrange(len(train_data))):
        count_total += 1
        tokens = train_data[index]['tokens']
        chars = train_data[index]['chars']
        tags = train_data[index]['tags']
        if parameters['data_type'] == 'genia_pos':
            tags = [np.random.choice(x) for x in tags]
        if parameters['caps_dim']:
            caps = train_data[index]['caps']
        if parameters['data_type'] == 'ner' and parameters['gaz_dim']:
            gaz = train_data[index]['gazetteers']
        if parameters['pos_dim']:
            postags = train_data[index]['postags']
        tokens = insert_singletons(tokens, singletons)
        chars_for, chars_rev, chars_pos = pad_sentence_tokens_chars(chars)
        inp = []
        if parameters['token_emb_dim']:
            inp.append(tokens)
        if parameters['char_emb_dim']:
            inp.append(chars_for)
            if parameters['char_bidirect']:
                inp.append(chars_rev)
            inp.append(chars_pos)
        if parameters['caps_dim']:
            inp.append(caps)
        if parameters['data_type'] == 'ner' and parameters['gaz_dim']:
            inp.append(gaz)
        if parameters['pos_dim']:
            inp.append(postags)
        inp.append(tags)
        new_cost = f_train(*inp)
        epoch_costs.append(new_cost)
        last_costs.append(new_cost)
        if i % 10 == 0 and i > 0 == 0:
            logger.info("%i, average since last time: %f" % (i, np.mean(last_costs)))
            last_costs = []
        if count_total % 1000 == 0:
            dev_score = evaluate(dev_sentences, dev_data, A_t)
            test_score = evaluate(test_sentences, test_data, A_t)
            logger.info("Score on dev: %.5f" % dev_score)
            logger.info("Score on test: %.5f" % test_score)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                logger.info("New best score on dev.")
                experiment.dump("Saving model...")
            if test_score > best_test_score:
                best_test_score = test_score
                logger.info("New best score on test.")
    logger.info("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))
