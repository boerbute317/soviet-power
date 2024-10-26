# -*- coding: utf-8 -*-
from typing import List

from bert4torch.tokenizers import Tokenizer


class MyTokenizer(Tokenizer):
    def _tokenize(self, text, pre_tokenize=True):
        return list(text)

    def tokenize(self, text: str, maxlen: int = None) -> List[str]:
        tokens = [self._token_translate.get(token) or token for token in self._tokenize(text)]

        has_start_token = self._token_start is not None
        has_end_token = self._token_end is not None
        if has_start_token:
            tokens.insert(0, self._token_start)
        if has_end_token:
            tokens.append(self._token_end)

        if maxlen is not None:
            si, ei = 0, None
            if has_start_token:
                maxlen = maxlen - 1
                si = 1
            if has_end_token:
                maxlen = maxlen - 1
                ei = -1
            tokens = tokens[si:ei]
            tokens = tokens[:maxlen]
            if has_start_token:
                tokens.insert(0, self._token_start)
            if has_end_token:
                tokens.append(self._token_end)
        return tokens
