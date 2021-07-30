import jieba
import sentencepiece as spm
class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        #BasicTokenizer的对应类就在下面
        #WordpieceTokenizer的定义在BasicTokenizer定义的下面
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self._vocab_size = self.sp_model.get_piece_size()
        self.inv_vocab = {}
        for  i  in  range(self._vocab_size):
            self.inv_vocab[i] = self.sp_model.id_to_piece(i)
        self.vocab = {v:k for k,v in self.inv_vocab.items()}

    def convert_tokens_to_ids(self,vocab,items):
        #由单词对应到相应的id内容
        output = []
        r"""
        切词语的方法：能找到词组的时候将对应的id放入数组之中
        找不到词组的时候如果能找到'_'+item将'_'+item放入数组之中
        如果什么都找不到的时候将所有单词对应的内容一一切分
        """
        for item in items:
            #print('item = ')
            #print(item)
            if vocab.__contains__(item):
                #print('situation1')
                #print(item)
                output.append(vocab[item])
                #字典中存在vocab[item],直接放入output数组
            else:
                #print('situation2')
                #print(item)
                currents = []
                while len(item) != 0:
                    for i in range(len(item)-1,-1,-1):
                        currentdata = item[0:i]
                        if vocab.__contains__(currentdata):
                            currents.append(vocab[currentdata])
                            item = item[i:]
                output.extend(currents)
        return output

    def convert_ids_to_tokens(self,items):
        #由id内容对应到相应的单词内容
        output = []
        for item in items:
            if self.inv_vocab[item] == '▁':
                output.append(' ')
            elif self.inv_vocab[item] == '\n':
                output.append('▂')
            elif self.inv_vocab[item][0] == '▁':
                output.append(self.inv_vocab[item][1:])
            else:
                output.append(self.inv_vocab[item])
        return output

    def tokenize(self, text):
        data = jieba.lcut(text)
        #data = ['我','是','绍兴人']
        i = 0
        while i < len(data):
            if data[i] != '\n' and data[i] != '▁':
                data[i] = '▁'+data[i]
            elif data[i] == '\n':
                data[i] = '▂'
            i = i+1
        print('***data = ***')
        print(data)
        #data = ['▁我', '▁是', '▁绍兴人']
        data = self.convert_tokens_to_ids(self.vocab,data)
        return data
        #data = ['我', '▁是', '小顾', '仔']
