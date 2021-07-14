import argparse
import sys
from collections import Counter
import string

bad_characters_set = set(string.digits + string.punctuation)


def is_good_token(tok):
    return bool(set(tok) - bad_characters_set)


def jaccard_coef(x, y):
    x_counter = Counter(filter(is_good_token, x))
    y_counter = Counter(filter(is_good_token, y))
    isect_cnt = sum((x_counter & y_counter).values())
    union_cnt = sum(x_counter.values()) + sum(y_counter.values()) - isect_cnt
    return 0.0 if not union_cnt else float(isect_cnt) / float(union_cnt)


class StreamNull(object):
    def write(self, *args, **kwargs):
        pass


def check_src_dst(line1, line2, args):
    words1 = line1.split(' ')
    words2 = line2.split(' ')

    # Filter by sentence length
    #
    len1 = len(words1)
    len2 = len(words2)
    if max(len1, len2) > args.max_sent_len:
        msg = 'sent_len: %i/%i' % (len1, len2)
        return (1, msg)

    # Filter by word length
    #
    wlen1 = max(map(len, words1))
    wlen2 = max(map(len, words2))
    if args.max_word_len and max(wlen1, wlen2) > args.max_word_len:
        msg = 'word_len: %i/%i' % (wlen1, wlen2)
        return (2, msg)

    # Filter by bad UTF
    #
    UTF_REPL = str(b'\xef\xbf\xbd', encoding="UTF-8")
    bad1 = UTF_REPL in line1
    bad2 = UTF_REPL in line2
    if args.no_bad_utf and (bad1 or bad2):
        msg = 'bad_utf: %s/%s' % (bad1, bad2)
        return (3, msg)

    # Filter by zero length
    #
    zero1 = (wlen1 == 0)
    zero2 = (wlen2 == 0)
    if args.no_zero_len and (zero1 or zero2):
        msg = 'zero_len: %s/%s' % (zero1, zero2)
        return (4, msg)

    # Filter by Jaccard coefficient
    if args.max_jaccard_coef_exclusive:
        coef = jaccard_coef(words1, words2)
        if coef >= args.max_jaccard_coef_exclusive:
            msg = 'jaccard_coef: %s >= %s' % (coef, args.max_jaccard_coef_exclusive)
            return (5, msg)

    # Filter by equality for short numerical strings
    if args.filter_equality:
        if line1 == line2:
            msg = 'Equal lines'
            return (6, msg)

    return (0, "")


def parse_args():
    msg = 'Filter out long parallel sentences'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("--folder", type=str, required=False,
                        help="the folder which contains the data")
    parser.add_argument('--directions', type=str, default=None, required=False)
    parser.add_argument('--max-sent-len', type=int, required=True, help='max sentence length')
    parser.add_argument('--max-word-len', type=int, required=False, help='max word length')
    parser.add_argument('--filter-equality', action='store_true', default=False,
                        help='filter out string based on == equality')
    parser.add_argument('--no-bad-utf', action='store_true', default=False, help='filter out bad utf')
    parser.add_argument('--no-zero-len', action='store_true', default=False, help='filter out zero-len sentences')
    parser.add_argument('--max-jaccard-coef-exclusive', type=float, required=False, default=0.0,
                        help='max jaccard coefficient between source and target sentences (exclusive, i.e. not accepting specified value)')
    parser.add_argument('--rejected', type=argparse.FileType('w'), default=StreamNull,
                        help='write rejected sentences here')
    args = parser.parse_args()
    return args


def deup(src_file, tgt_file, src_file_out, tgt_file_out):
    seen = set()
    dup_count = 0
    with open(src_file, encoding='utf-8') as fsrc, \
            open(tgt_file, encoding='utf-8') as ftgt, \
            open(src_file_out, 'w', encoding='utf-8') as fsrc_out, \
            open(tgt_file_out, 'w', encoding='utf-8') as ftgt_out:
        for s, t in zip(fsrc, ftgt):
            if (s, t) not in seen:
                fsrc_out.write(s)
                ftgt_out.write(t)
                seen.add((s, t))
            else:
                dup_count += 1
    print(f'number of duplication: {dup_count}')


def main():
    args = parse_args()

    seen = set()
    for line in sys.stdin:
        line = line.rstrip('\n')
        parts = line.split('\t')

        status, msg = check_src_dst(parts[0], parts[1], args)

        if status == 0:
            if (parts[0], parts[1]) not in seen:
                print("\t".join(parts))
                seen.add((parts[0], parts[1]))
        else:
            print("\t".join([msg, parts[0], parts[1]]), file=args.rejected)


if __name__ == '__main__':
    main()