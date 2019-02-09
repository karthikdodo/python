name = input('enter sequence of gen')
set2 = [name[i:i+3] for i in range(0,len(name)-2)]
file2 = open('codon.tsv')
print(set2)
counts = dict()
for line in file2:
  if any (word in (line.split()[0]) for word in set2):
      print((line.split()[1]))
      counts[line.split()[1]] = counts.get(line.split()[1],0)+1
      print(counts)