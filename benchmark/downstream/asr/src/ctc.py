import numpy as np


class CTCPrefixScore():
    ''' 
    CTC Prefix score calculator
    An implementation of Algo. 2 in https://www.merl.com/publications/docs/TR2017-190.pdf (Watanabe et. al.)
    Reference (official implementation): https://github.com/espnet/espnet/tree/master/espnet/nets
    '''

    def __init__(self, x):
        self.logzero = -100000000.0
        self.blank = 0
        self.eos = 1
        self.x = x.cpu().numpy()[0]
        self.odim = x.shape[-1]
        self.input_length = len(self.x)

    def init_state(self):
        # 0 = non-blank, 1 = blank
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)

        # Accumalate blank at each step
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i-1, 1] + self.x[i, self.blank]
        return r

    def full_compute(self, g, r_prev):
        '''Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function computes all possible tokens for c (memory inefficient)'''
        prefix_length = len(g)
        last_char = g[-1] if prefix_length > 0 else 0

        # init. r
        r = np.full((self.input_length, 2, self.odim),
                    self.logzero, dtype=np.float32)

        # start from len(g) because is impossible for CTC to generate |y|>|X|
        start = max(1, prefix_length)

        if prefix_length == 0:
            r[0, 0, :] = self.x[0, :]    # if g = <sos>

        psi = r[start-1, 0, :]

        phi = np.logaddexp(r_prev[:, 0], r_prev[:, 1])

        for t in range(start, self.input_length):
            # prev_blank
            prev_blank = np.full((self.odim), r_prev[t-1, 1], dtype=np.float32)
            # prev_nonblank
            prev_nonblank = np.full(
                (self.odim), r_prev[t-1, 0], dtype=np.float32)
            prev_nonblank[last_char] = self.logzero

            phi = np.logaddexp(prev_nonblank, prev_blank)
            # P(h|current step is non-blank) = [ P(prev. step = y) + P()]*P(c)
            r[t, 0, :] = np.logaddexp(r[t-1, 0, :], phi) + self.x[t, :]
            # P(h|current step is blank) = [P(prev. step is blank) + P(prev. step is non-blank)]*P(now=blank)
            r[t, 1, :] = np.logaddexp(
                r[t-1, 1, :], r[t-1, 0, :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi+self.x[t, :])

        #psi[self.eos] = np.logaddexp(r_prev[-1,0], r_prev[-1,1])
        return psi, np.rollaxis(r, 2)

    def cheap_compute(self, g, r_prev, candidates):
        '''Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function considers only those tokens in candidates for c (memory efficient)'''
        prefix_length = len(g)
        odim = len(candidates)
        last_char = g[-1] if prefix_length > 0 else 0

        # init. r
        r = np.full((self.input_length, 2, len(candidates)),
                    self.logzero, dtype=np.float32)

        # start from len(g) because is impossible for CTC to generate |y|>|X|
        start = max(1, prefix_length)

        if prefix_length == 0:
            r[0, 0, :] = self.x[0, candidates]    # if g = <sos>

        psi = r[start-1, 0, :]
        # Phi = (prev_nonblank,prev_blank)
        sum_prev = np.logaddexp(r_prev[:, 0], r_prev[:, 1])
        phi = np.repeat(sum_prev[..., None],odim,axis=-1)
        # Handle edge case : last tok of prefix in candidates
        if  prefix_length>0 and last_char in candidates:
            phi[:,candidates.index(last_char)] = r_prev[:,1]

        for t in range(start, self.input_length):
            # prev_blank
            # prev_blank = np.full((odim), r_prev[t-1, 1], dtype=np.float32)
            # prev_nonblank
            # prev_nonblank = np.full((odim), r_prev[t-1, 0], dtype=np.float32)
            # phi = np.logaddexp(prev_nonblank, prev_blank)
            # P(h|current step is non-blank) =  P(prev. step = y)*P(c)
            r[t, 0, :] = np.logaddexp( r[t-1, 0, :], phi[t-1]) + self.x[t, candidates]
            # P(h|current step is blank) = [P(prev. step is blank) + P(prev. step is non-blank)]*P(now=blank)
            r[t, 1, :] = np.logaddexp( r[t-1, 1, :], r[t-1, 0, :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi[t-1,]+self.x[t, candidates])

        # P(end of sentence) = P(g)
        if self.eos in candidates:
            psi[candidates.index(self.eos)] = sum_prev[-1]
        return psi, np.rollaxis(r, 2)
