import json
from collections import Counter

d = json.load(open('data/generated/train/metadata.json'))

# Get all unique answers
answers = [s['answer'] for s in d]
answer_counts = Counter(answers)

print(f'Total samples: {len(d)}')
print(f'Unique answers: {len(answer_counts)}')
print('\nTop 30 most common answers:')
for ans, count in answer_counts.most_common(30):
    print(f'  "{ans}": {count}')

print('\nQuestion types:')
types = Counter(s['question_type'] for s in d)
for t, c in types.items():
    print(f'  {t}: {c}')
