import matplotlib.pyplot as plt
import numpy as np

results = []

for i in range(6):
    data = []
    with open(f'DragLift_case_{i+1}.fce', 'r') as f:
        lines = f.readlines()
        for line in lines[5:]:
            items = line.strip().split()
            items = [float(item) for item in items]
            data.append(items)
    data = np.array(data)
    results.append(data)

start=100
plt.figure(figsize=(5,4))
for i in range(6):
    plt.plot(results[i][start:,0], results[i][start:,6], label=f'case_{i+1}')
plt.grid()
# plt.title('Lift plots')
plt.legend()
plt.xlabel('time $c/U_{\inf}$')
plt.ylabel('force dimensionless')
plt.tight_layout()
plt.savefig('lift_convergence.png', dpi=400)
plt.close()

plt.figure(figsize=(5,4))
for i in range(6):
    plt.plot(results[i][start:,0], results[i][start:,3], label=f'case_{i+1}')
plt.grid()
# plt.title('Drag plots')
plt.legend()
plt.xlabel('time $c/U_{\inf}$')
plt.ylabel('force dimensionless')
plt.tight_layout()
plt.savefig('drag_convergence.png', dpi=400)
plt.close()


# Steady forces
for i in range(6): # lift
    print('%.3f' % results[i][-1,6], end=' & ')
print()
for i in range(6): # drag
    print('%.3f' % results[i][-1,3], end=' & ')
print()
# Steady forces percentage errors
for i in range(6): # lift
    print('%.3f' % np.abs(100*(results[i][-1,6]-results[6][-1,6])/results[6][-1,6]), end='% & ')
print()
for i in range(6): # drag
    print('%.3f' % np.abs(100*(results[i][-1,3]-results[6][-1,3])/results[6][-1,3]), end='% & ')
print()