#!/usr/bin/env python
# coding: utf-8

# In[76]:


def triangular_sum(n):
    sum = 0
    for i in range(n + 1):
        sum += i
    return sum

def rc_to_pos(row, col, n):
    return (row - 1) * n + col

def pos_to_rc(p, n):
    row = int((p - 1) / n) + 1
    col = p % n
    if(col == 0):
        col = n
    return [row, col]

def block_by_position(blocks, row, col, n):
    b = blocks[0]
    for block in blocks:
        pos = block[2]
        for p in pos:
            if p == rc_to_pos(row, col, n):
                b = block
    return b

def same_rc(s1, s2, n):
    bool = False
    row1 = int((s1 - 1) / n)
    row2 = int((s2 - 1) / n)
    col1 = s1 % n
    col2 = s2 % n
    if(row1 == row2 or col1 == col2):
        bool = True
    return bool

def same_rc_set(set, n):
    bool_r = False
    bool_c = False
    for i in range(len(set)):
        for j in range(len(set)):
            pos_i = pos_to_rc(set[i], n)
            pos_j = pos_to_rc(set[j], n)
            if(i != j and pos_i[0] == pos_j[0]):
                bool_r = True
            if(i != j and pos_i[1] == pos_j[1]):
                bool_c = True
    return [bool_r, bool_c]

def rows_cols(probs):
    temp = []
    big_n = len(probs)
    n = int(big_n ** 0.5)
    rows = []
    cols = []
    for i in range(1, n + 1):
        r = []
        c = []
        for j in range(1, n + 1):
            r.append((i - 1) * n + j)
            c.append((j - 1) * n + i)
        rows.append(r)
        cols.append(c)
    return [rows, cols]

def sum_to_get(vals, num):
    bool = False
    sum = 0
    for val in vals:
        sum += val
    if(sum == num):
        bool = True
    return bool

def mul_to_get(vals, num):
    bool = False
    prod = 1
    for val in vals:
        prod *= val
    if(prod == num):
        bool = True
    return bool

def comb_vals(t, num, m, n):
    temp = []
    if(t == 'a'):
        if(m == 2):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i + j == num:
                        temp.append([i, j])
        if(m > 2):
            for i in range(1, n + 1):
                if(num - i  > 0):
                    old = comb_vals(t, num - i, m-1, n)
                    for val in old:
                        lis = [i]
                        for value in val:
                            lis.append(value)
                        if(lis not in temp):
                            temp.append(lis)
    if(t == 's'):
        if(m == 2):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if abs(i - j) == num:
                        temp.append([i, j])
        if(m > 2):
            for i in range(1, n + 1):
                old_a = comb_vals('a', abs(num - i), m-1, n)
                for val in old_a:
                    lis = [i]
                    for value in val:
                        lis.append(value)
                    if(sum_to_get(lis, num) == False and lis not in temp):
                        temp.append(lis)
                old_s = comb_vals('s', num + i, m-1, n)
                for val in old_s:
                    lis = [i]
                    for value in val:
                        lis.append(value)
                    if(lis not in temp):
                        temp.append(lis)
    if(t == 'm'):
        if(m == 2):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i * j == num:
                        temp.append([i, j])
        if(m > 2):
            for i in range(1, n + 1):
                if(num % i  == 0):
                    old = comb_vals(t, num / i, m-1, n)
                    for val in old:
                        lis = [i]
                        for value in val:
                            lis.append(value)
                        if(lis not in temp):
                            temp.append(lis)
    if(t == 'd'):
        if(m == 2):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if max(i, j) / min(i, j) == num:
                        temp.append([i, j])
        if(m > 2):
            for i in range(1, n + 1):
                maxim = max(i, num)
                minim = min(i, num)
                divis = maxim / minim
                old_m = comb_vals('m', divis, m-1, n)
                for val in old_m:
                    lis = [i]
                    for value in val:
                        lis.append(value)
                    if(sum_to_get(lis, num) == False and lis not in temp):
                        temp.append(lis)
                old_d = comb_vals('d', num * i, m-1, n)
                for val in old_d:
                    lis = [i]
                    for value in val:
                        lis.append(value)
                    if(lis not in temp):
                        temp.append(lis)
    if(t == 'g'):
        temp.append([num])
    return temp

def combinations(b, n):
    t, num, s = b[0], b[1], b[2]
    temp = []
    l = len(s)
    vals = comb_vals(t, num, l, n)
    for val in vals:
        bool = True
        for i in range(l):
            for j in range(i + 1, l):
                if(val[i] == val[j] and same_rc(s[i], s[j], n)):
                    bool = False
        if(bool == True):
            temp.append(val)
    return [temp, s]

def all_combs(b, n):
    temp = []
    combs = []
    num, s = b[0], b[1]
    if(len(s) == 1):
        return combinations(['g', num, s], n)
    set1 = ['a', num, s]
    set2 = ['m', num, s]
    sets = [set1, set2]
    if(len(s) < 3):
        set3 = ['s', num, s]
        set4 = ['d', num, s]
        sets.append(set3)
        sets.append(set4)
    for block in sets:
        combs.append(combinations(block, n)[0])
    for comb in combs:
        for combo in comb:
            if(combo not in temp and len(comb) > 0):
                temp.append(combo)
    return [temp, s]

def combination_set(blocks, n):
    temp = []
    if(len(blocks[0]) == 3):
        for block in blocks:
            temp.append(combinations(block, n))
    if(len(blocks[0]) == 2):
        for block in blocks:
            temp.append(all_combs(block, n))
    return temp

def all_add_comb(comb):
    temp = []
    combos = comb[0]
    for combo in combos:
        sum = 0
        for c in combo:
            sum += c
        if(sum not in temp):
            temp.append(sum)
    return temp

def all_add_probs(probs):
    temp = []
    
    return temp

def add_lists_to(lists, num):
    temp = []
    n = len(lists)
    main_list = lists[0]
    if(n == 2):
        other_list = lists[1]
        for m in main_list:
            for o in other_list:
                if(m + o == num):
                    temp.append([m, o])
    else:
        old = []
        for i in range(1, n):
            old.append(lists[i])
        for m in main_list:
            new_num = num - m
            old_set = add_lists_to(old, new_num)
            for s in old_set:
                temp_list = [m]
                for l in s:
                    temp_list.append(l)
                temp.append(temp_list)
    return temp

def comb_of_pos(combs, p):
    block, place = 0, 0
    pos = []
    for comb in combs:
        pos.append(comb[1])
    l_i = len(pos)
    for i in range(l_i):
        posit = pos[i]
        l_j = len(posit)
        for j in range(l_j):
            if(posit[j] == p):
                block = i
                place = j
    return block, place

def block_of_pos(blocks, p):
    block, place = 0, 0
    pos = []
    for b in blocks:
        pos.append(b[2])
    l_i = len(pos)
    for i in range(l_i):
        posit = pos[i]
        l_j = len(posit)
        for j in range(l_j):
            if(posit[j] == p):
                block = i
                place = j
    return block, place
    
def quant_pos(combs, p, n):
    pos_list = []
    comb_list = []
    desired_i = 0
    desired_j = 0
    for comb in combs:
        pos_list.append(comb[1])
        comb_list.append(comb[0])
    l_i = len(pos_list)
    for i in range(l_i):
        l_j = len(pos_list[i])
        for j in range(l_j):
            if pos_list[i][j] == p:
                desired_i = i
                desired_j = j
    block = combs[desired_i][0]
    posibilities = []
    for comb in block:
        posibilities.append(comb[desired_j])
    return sorted(set(posibilities))


# In[75]:


def prob_to_comb(combs, probs):
    temp = []
    n = len(probs)
    for comb in combs:
        coms = comb[0]
        pos = comb[1]
        temp_c = []
        for com in coms:
            bool = True
            for p in range(len(pos)):
                if(com[p] not in probs[pos[p] - 1]):
                    bool = False
            if(bool):
                temp_c.append(com)
        temp.append([temp_c, pos])
    rows, cols = rows_cols(probs)[0], rows_cols(probs)[1]
    temp = rc_block_lock(temp, probs, rows, cols)
    return temp

def comb_to_prob(combs, n):
    temp = []
    small_n = int(n ** 0.5)
    for p in range(1, n + 1):
        temp.append(quant_pos(combs, p, n))
    rows, cols = rows_cols(temp)[0], rows_cols(temp)[1]
    temp = rc_lock(temp, rows, cols)
    temp = tyrant_lock(temp, rows, cols)
    temp = duo_lock(temp, rows, cols)
    temp = block_rc_lock(combs, temp, rows, cols)
    temp = add_lock(combs, temp, rows, cols)
    return temp


# In[55]:


def rc_lock(probs, rows, cols):
    big_n = len(probs)
    n = int(big_n ** 0.5)
    temp_r = []
    temp_c = []
    for row in rows:
        row_probs = []
        locks = []
        for m in range(1, n + 1):
            for i in row:
                prob_i = probs[i - 1]
                if(len(prob_i) == m):
                    for_lock = 1
                    for j in row:
                        prob_j = probs[j - 1]
                        if(j != i and set(prob_i) == set(prob_i).union(set(prob_j))):
                            for_lock += 1
                    if(for_lock == m):
                        locks.append(prob_i)
        for r in row:
            prob = probs[r - 1]
            for lock in locks:
                if(set(prob) != set(prob).intersection(set(lock))):
                        prob = [k for k in prob if k not in lock]
            row_probs.append(prob)
        temp_r.append(row_probs)
    for col in cols:
        col_probs = []
        locks = []
        for m in range(1, n + 1):
            for i in col:
                prob_i = probs[i - 1]
                if(len(prob_i) == m):
                    for_lock = 1
                    for j in col:
                        prob_j = probs[j - 1]
                        if(j != i and set(prob_i) == set(prob_i).union(set(prob_j))):
                            for_lock += 1
                    if(for_lock == m):
                        locks.append(prob_i)
        for c in col:
            prob = probs[c - 1]
            for lock in locks:
                if(set(prob) != set(prob).intersection(set(lock))):
                        prob = [k for k in prob if k not in lock]
            col_probs.append(prob)
        temp_c.append(col_probs)
    return superposition(temp_r, temp_c)

def tyrant_lock(probs, rows, cols):
    big_n = len(probs)
    n = int(big_n ** 0.5)
    temp_r = []
    temp_c = []
    for row in rows:
        row_probs = []
        for r in row:
            bool = True
            prob = probs[r - 1]
            num = 0
            for p in prob:
                count = 1
                for i in row:
                    if(i != r and p in probs[i - 1]):
                        count += 1
                if(count == 1):
                    bool = False
                    num = p
            if(bool):
                row_probs.append(prob)
            if(bool == False):
                row_probs.append([num])
        temp_r.append(row_probs)
    for col in cols:
        col_probs = []
        for c in col:
            bool = True
            prob = probs[c - 1]
            num = 0
            for p in prob:
                count = 1
                for i in col:
                    if(i != c and p in probs[i - 1]):
                        count += 1
                if(count == 1):
                    bool = False
                    num = p
            if(bool):
                col_probs.append(prob)
            if(bool == False):
                col_probs.append([num])
        temp_c.append(col_probs)
    return superposition(temp_r, temp_c)

def duo_lock(probs, rows, cols):
    big_n = len(probs)
    n = int(big_n ** 0.5)
    temp_r = []
    temp_c = []
    locks = []
    for row_i in rows:
        r_num_i = pos_to_rc(row_i[0], n)[0]
        for m in range(1, n + 1):
            list_r = []
            list_c = []
            for r_i in row_i:
                c_num_i = pos_to_rc(r_i, n)[1]
                if(m in probs[r_i - 1]):
                    list_c.append(c_num_i)
            if(len(list_c) == 2):
                list_r.append(r_num_i)
                for row_j in rows:
                    r_num_j = pos_to_rc(row_j[0], n)[0]
                    if(row_j > row_i):
                        c_j = []
                        for r_j in row_j:
                            c_num_j = pos_to_rc(r_j, n)[1]
                            if(m in probs[r_j - 1]):
                                c_j.append(c_num_j)
                        if(c_j == list_c):
                            list_r.append(r_num_j)
            if(len(list_r) == 2):
                locks.append([m, list_r, list_c])
    for col_i in cols:
        c_num_i = pos_to_rc(col_i[0], n)[1]
        for m in range(1, n + 1):
            list_r = []
            list_c = []
            for c_i in col_i:
                r_num_i = pos_to_rc(c_i, n)[0]
                if(m in probs[c_i - 1]):
                    list_r.append(r_num_i)
            if(len(list_r) == 2):
                list_c.append(c_num_i)
                for col_j in cols:
                    c_num_j = pos_to_rc(col_j[0], n)[1]
                    if(col_j > col_i):
                        r_j = []
                        for c_j in col_j:
                            r_num_j = pos_to_rc(c_j, n)[0]
                            if(m in probs[c_j - 1]):
                                r_j.append(r_num_j)
                        if(r_j == list_r):
                            list_c.append(c_num_j)
            if(len(list_c) == 2):
                locks.append([m, list_r, list_c])
    for row in rows:
        row_probs = []
        for r in row:
            rc = pos_to_rc(r, n)
            prob = probs[r - 1]
            new_prob = []
            for p in prob:
                bool = True
                for lock in locks:
                    if(p == lock[0] and (rc[0] in lock[1] or rc[1] in lock[2])):
                        bool = False
                        if(rc[0] in lock[1] and rc[1] in lock[2]):
                            bool = True
                if(bool):
                    new_prob.append(p)
            row_probs.append(new_prob)
        temp_r.append(row_probs)
    for col in cols:
        col_probs = []
        for c in col:
            rc = pos_to_rc(c, n)
            prob = probs[c - 1]
            new_prob = []
            for p in prob:
                bool = True
                for lock in locks:
                    if(p == lock[0] and (rc[0] in lock[1] or rc[1] in lock[2])):
                        bool = False
                        if(rc[0] in lock[1] and rc[1] in lock[2]):
                            bool = True
                if(bool):
                    new_prob.append(p)
            col_probs.append(new_prob)
        temp_c.append(col_probs)
    return superposition(temp_r, temp_c)

def block_rc_lock(combs, probs, rows, cols):
    big_n = len(probs)
    n = int(big_n ** 0.5)
    temp_r = []
    temp_c = []
    row_locks = []
    col_locks = []
    for comb in combs:
        combos = comb[0]
        posit = comb[1]
        for m in range(1, n + 1):
            bool = True
            pos_of_m = []
            for combo in combos:
                if(m in combo):
                    for i in range(len(combo)):
                        if combo[i] == m and posit[i] not in pos_of_m:
                            pos_of_m.append(posit[i])
                if(m not in combo):
                    bool = False
            if(bool == True and len(pos_of_m) > 0):
                if same_rc_set(pos_of_m, n)[1] == False:
                    row_locks.append([m, pos_to_rc(pos_of_m[0], n)[0], posit])
                if same_rc_set(pos_of_m, n)[0] == False:
                    col_locks.append([m, pos_to_rc(pos_of_m[0], n)[1], posit])
    for row in rows:
        row_probs = []
        for r in row:
            rc = pos_to_rc(r, n)
            prob = probs[r - 1]
            new_prob = []
            for p in prob:
                bool = True
                for lock in row_locks:
                    if(p == lock[0] and rc[0] == lock[1] and r not in lock[2]):
                        bool = False
                if(bool):
                    new_prob.append(p)
            row_probs.append(new_prob)
        temp_r.append(row_probs)
    for col in cols:
        col_probs = []
        for c in col:
            rc = pos_to_rc(c, n)
            prob = probs[c - 1]
            new_prob = []
            for p in prob:
                bool = True
                for lock in col_locks:
                    if(p == lock[0] and rc[1] == lock[1] and c not in lock[2]):
                        bool = False
                if(bool):
                    new_prob.append(p)
            col_probs.append(new_prob)
        temp_c.append(col_probs)
    return superposition(temp_r, temp_c)

def rc_block_lock(combs, probs, rows, cols):
    big_n = len(probs)
    n = int(big_n ** 0.5)
    temp_combs = []
    row_locks = []
    col_locks = []
    for row in rows:
        for m in range(1, n + 1):
            block_of_m = []
            for r in row:
                if(m in probs[r - 1] and comb_of_pos(combs, r)[0] not in block_of_m):
                    block_of_m.append(comb_of_pos(combs, r)[0])
            if(len(block_of_m) == 1):
                row_locks.append([m, combs[block_of_m[0]], row])
    for col in cols:
        for m in range(1, n + 1):
            block_of_m = []
            for c in col:
                if(m in probs[c - 1] and comb_of_pos(combs, c)[0] not in block_of_m):
                    block_of_m.append(comb_of_pos(combs, c)[0])
            if(len(block_of_m) == 1):
                col_locks.append([m, combs[block_of_m[0]], col])
    for comb in combs:
        new_combos = []
        combos = comb[0]
        posit = comb[1]
        for combo in combos:
            bool = True
            for lock in row_locks:
                if(lock[0] not in combo and lock[1] == comb):
                    bool = False
                if(lock[0] in combo and lock[1] == comb):
                    counter = 0
                    for pos in posit:
                        i = posit.index(pos)
                        if(combo[i] == lock[0] and pos in lock[2]):
                            counter += 1
                    if(counter == 0):
                        bool = False
            for lock in col_locks:
                if(lock[0] not in combo and lock[1] == comb):
                    bool = False
                if(lock[0] in combo and lock[1] == comb):
                    counter = 0
                    for pos in posit:
                        i = posit.index(pos)
                        if(combo[i] == lock[0] and pos in lock[2]):
                            counter += 1
                    if(counter == 0):
                        bool = False
            if(bool):
                new_combos.append(combo)
        temp_combs.append([new_combos, posit])
    return temp_combs

def add_lock(combs, probs, rows, cols):
    big_n = len(probs)
    n = int(big_n ** 0.5)
    add_to = triangular_sum(n)
    temp = []
    locks = []
    for i in range(n):
        temp_rows = []
        for j in range(3):
            if(i + j < n):
                temp_rows.append(rows[i + j])
        for k in range(1, len(temp_rows) + 1):
            temp_temp_rows = []
            for r in range(k):
                temp_temp_rows.append(temp_rows[r])
            positions_r = []
            for row in temp_temp_rows:
                for pos in row:
                    positions_r.append(pos)
            temp_sum = k * add_to
            in_comb = []
            not_in_comb = []
            for comb in combs:
                bool = True
                posit = comb[1]
                for pos in posit:
                    if pos not in positions_r:
                        bool = False
                if(bool == True):
                    in_comb.append([posit, all_add_comb(comb)])
                if(bool == False):
                    for pos in posit:
                        if(pos in positions_r):
                            not_in_comb.append([pos, quant_pos(combs, pos, n)])
            if(len(not_in_comb) == 1):
                N = len(in_comb)
                set_to_add = []
                for set in in_comb:
                    set_to_add.append(set[1])
                for set in not_in_comb:
                    set_to_add.append(set[1])
                sum_posibilities = add_lists_to(set_to_add, temp_sum)
                locks.append([not_in_comb[0][0], sum_posibilities[0][N]])
    for i in range(n):
        temp_cols = []
        for j in range(3):
            if(i + j < n):
                temp_cols.append(cols[i + j])
        for k in range(1, len(temp_cols) + 1):
            temp_temp_cols = []
            for c in range(k):
                temp_temp_cols.append(temp_cols[c])
            positions_c = []
            for col in temp_temp_cols:
                for pos in col:
                    positions_c.append(pos)
            temp_sum = k * add_to
            in_comb = []
            not_in_comb = []
            for comb in combs:
                bool = True
                posit = comb[1]
                for pos in posit:
                    if pos not in positions_r:
                        bool = False
                if(bool == True):
                    in_comb.append([posit, all_add_comb(comb)])
                if(bool == False):
                    for pos in posit:
                        if(pos in positions_c):
                            not_in_comb.append([pos, quant_pos(combs, pos, n)])
            if(len(not_in_comb) == 1):
                N = len(in_comb)
                set_to_add = []
                for set in in_comb:
                    set_to_add.append(set[1])
                for set in not_in_comb:
                    set_to_add.append(set[1])
                sum_posibilities = add_lists_to(set_to_add, temp_sum)
                locks.append([not_in_comb[0][0], sum_posibilities[0][N]])
    for p in range(1, big_n + 1):
        prob = probs[p - 1]
        bool = True
        for lock in locks:
            pos = lock[0]
            pro = lock[1]
            if(pos == p):
                bool = False
                temp.append([pro])
        if(bool):
            temp.append(prob)
    return temp

def even_odd_lock(blocks, probs, rows, cols):
    temp = []
    return temp

def superposition(temp_r, temp_c):
    temp = []
    n = len(temp_r)
    big_n = int(n ** 2)
    for p in range(1, big_n + 1):
        r, c = pos_to_rc(p, n)[0], pos_to_rc(p, n)[1]
        temp.append(list(set(temp_r[r - 1][c - 1]).intersection(set(temp_c[c - 1][r - 1]))))
    return temp

test_combs = combination_set(puzzle_69420, 9)
test_probs = comb_to_prob(test_combs, 81)

test_rows = rows_cols(test_probs)[0]
test_cols = rows_cols(test_probs)[1]

print_posibilities(test_probs, 81, 9)
print_posibilities(add_lock(test_combs, test_probs, test_rows, test_cols), 81, 9)


# In[15]:


def generic_puzzle(n, puzzle):
    list_of_zeros = []
    for block in puzzle:
        positions = block[len(block) - 1]
        for position in positions:
            list_of_zeros.append(position)
    string = "["
    for i in range(n):
        string += '['
        for j in range(1, n + 1):
            num = i * n + j
            if(num in list_of_zeros):
                string += ' '
            else:
                string += str(i * n + j)
            if(j < n):
                string += ', '
            if(num < 10 and j < n):
                string += ' '
            if(i > 0 and j < n and num in list_of_zeros):
                string += ' '
            if(j == n and i < n - 1):
                string += '],\n'
    string += "]]\n,\n    ['', , []]"
    return string

def print_puzzle(probs):
    big_n = len(probs)
    n = big_n ** 0.5
    string = ""
    for i in range(big_n):
        if(i % n == 0):
            string += "\n"
        p = probs[i]
        if(len(p) == 1):
            string += str(p[0]) + " "
        else:
            string += "  "
    print(string)
    
def print_posibilities(p, n, small_n):
    string = ""
    for i in range(n):
        string += str(p[i]) + " "
        if((i + 1) % small_n == 0):
            string += "\n"
    print(string)
        
def puzzle_size(blocks):
    size = 0
    spaces = []
    for block in blocks:
        spaces.append(block[len(block) - 1])
    for space in spaces:
        size += len(space)
    return size

def iteration(combs, probs, n):
    pc = prob_to_comb(combs, probs)
    cp = comb_to_prob(pc, n)
    return [pc, cp]

def iterator(blocks, N):
    n = puzzle_size(blocks)
    small_n = int(n ** 0.5)
    c = combination_set(blocks, small_n)
    p = comb_to_prob(c, n)
    for i in range(N):
        iter = iteration(c, p, n)
        c = iter[0]
        p = iter[1]
    print_puzzle(p)
    print_posibilities(p, n, small_n)


# In[52]:


puzzle_38818 = [
    ['a', 5, [1, 4]],
    ['m', 3, [2, 3, 6]],
    ['a', 6, [5, 7, 8]],
    ['g', 2, [9]]
]
    
iterator(puzzle_38818, 1)


# In[59]:


expert_4 = [
    ['m', 12, [1, 2]],
    ['a', 4, [5, 6]],
    ['d', 2, [3, 7]],
    ['d', 2, [4, 8]],
    ['d', 2, [9, 13]],
    ['s', 1, [10, 11]],
    ['a', 5, [14, 15]],
    ['s', 2, [12, 16]]
]

iterator(expert_4, 3)


# In[60]:


puzzle_91590 = [
    ['a', 6, [1, 6]],
    ['a', 3, [2, 3]],
    ['a', 11, [4, 5, 9]],
    ['a', 8, [7, 8]],
    ['a', 3, [10, 15]],
    ['a', 10, [11, 12, 16]],
    ['a', 9, [13, 14, 18]],
    ['a', 7, [17, 21, 22]],
    ['a', 9, [19, 20, 25]],
    ['a', 9, [23, 24]]
]

iterator(puzzle_91590, 0)
iterator(puzzle_91590, 1)
iterator(puzzle_91590, 2)


# In[19]:


no_op_6 = [
    [5, [1]],
    [3, [2, 3]],
    [30, [4, 10]],
    [16, [5, 6, 12]],
    [216, [7, 8, 13, 14]],
    [1, [9]],
    [3, [15, 16]],
    [7, [11, 17]],
    [4, [18, 24]],
    [108, [29, 30, 36]],
    [20, [31, 32]],
    [6, [33, 34, 35]],
    [3, [19, 25]],
    [1, [20, 26]],
    [11, [21, 27]],
    [11, [22, 23, 28]]
    
]

iterator(no_op_6, 2)


# In[20]:


puzzle_63168 = [
    ['a', 9, [1, 8, 15]],
    ['s', 3, [2, 9]],
    ['a', 9, [3, 10]],
    ['a', 17, [4, 5, 6, 13]],
    ['s', 4, [11, 12]],
    ['a', 6, [7, 14]],
    ['a', 17, [16, 22, 23]],
    ['s', 1, [17, 24]],
    ['a', 5, [18, 25]],
    ['a', 15, [19, 20, 21]],
    ['s', 6, [29, 36]],
    ['a', 13, [30, 31, 32]],
    ['a', 9, [37, 43, 44]],
    ['a', 13, [38, 39]],
    ['s', 4, [45, 46]],
    ['a', 10, [26, 33, 40, 47]],
    ['g', 2, [27]],
    ['a', 14, [28, 34, 35]],
    ['s', 1, [41, 48]],
    ['a', 8, [42, 49]]
    
]

iterator(puzzle_63168, 5)


# In[21]:


puzzle_26603 = [
    ['m', 15, [1, 2]],
    ['m', 48, [3, 11]],
    ['g', 8, [9]],
    ['m', 42, [4, 5, 6]],
    ['d', 2, [7, 8]],
    ['m', 28, [17, 25]],
    ['m', 336, [10, 18, 26]],
    ['m', 5, [19, 27]],
    ['d', 3, [12, 20]],
    ['m', 20, [13, 21]],
    ['d', 3, [14, 15]],
    ['m', 12, [22, 23]],
    ['m', 10, [16, 24]],
    ['d', 2, [33, 41]],
    ['d', 2, [49, 57]],
    ['d', 4, [34, 35]],
    ['g', 5, [36]],
    ['m', 20, [42, 50]],
    ['m', 6, [43, 51]],
    ['m', 14, [58, 59]],
    ['m', 224, [44, 52, 60]],
    ['m', 48, [28, 29, 37]],
    ['m', 14, [30, 31]],
    ['g', 1, [32]],
    ['m', 56, [38, 39]],
    ['m', 21, [40, 48]],
    ['m', 48, [56, 64]],
    ['m', 30, [45, 46, 47]],
    ['d', 3, [53, 61]],
    ['m', 32, [54, 55]],
    ['m', 30, [62, 63]]
    
]

iterator(puzzle_26603, 2)


# In[61]:


puzzle_73454 = [
    ['m', 105, [1, 2, 3]],
    ['a', 20, [4, 12, 13]],
    ['d', 3, [5, 6]],
    ['s', 4, [7, 16]],
    ['s', 8, [8, 9]],
    ['d', 2, [10, 11]],
    ['s', 2, [14, 15]],
    ['m', 270, [17, 18, 27]],
    ['m', 216, [19, 20, 29]],
    ['s', 7, [21, 30]],
    ['d', 4, [22, 23]],
    ['m', 21, [24, 25]],
    ['a', 6, [26, 35, 44]],
    ['s', 3, [28, 37]],
    ['a', 15, [31, 32, 33]],
    ['a', 7, [34, 43]],
    ['a', 19, [36, 45, 54]],
    ['s', 1, [38, 39]],
    ['s', 6, [40, 49]],
    ['m', 63, [41, 42]],
    ['s', 5, [46, 55]],
    ['s', 6, [47, 56]],
    ['d', 2, [48, 57]],
    ['a', 16, [50, 58, 59]],
    ['m', 360, [51, 52, 53, 61]],
    ['s', 3, [60, 69]],
    ['m', 84, [62, 63, 72]],
    ['s', 2, [64, 73]],
    ['s', 4, [65, 74]],
    ['a', 6, [66, 67, 68]],
    ['m', 28, [70, 71]],
    ['s', 1, [75, 76]],
    ['g', 4, [77]],
    ['d', 3, [78, 79]],
    ['m', 15, [80, 81]]
]

iterator(puzzle_73454, 5)


# In[23]:


puzzle_207034 = [
    ['m', 105, [1, 2, 11]],
    ['s', 1, [3, 12]],
    ['d', 2, [4, 13]],
    ['m', 168, [5, 14, 23]],
    ['s', 8, [6, 15]],
    ['s', 1, [7, 16]],
    ['m', 160, [8, 9, 17]],
    ['a', 18, [10, 19, 28]],
    ['a', 17, [20, 29]],
    ['a', 13, [21, 22, 31]],
    ['a', 17, [18, 27, 36]],
    ['g', 9, [30]],
    ['a', 14, [24, 25, 32, 33]],
    ['a', 10, [26, 34, 35]],
    ['m', 126, [37, 38, 39]],
    ['m', 90, [40, 41, 42]],
    ['s', 4, [43, 44]],
    ['g', 1, [45]],
    ['m', 90, [46, 55, 64, 73]],
    ['d', 2, [47, 48]],
    ['m', 90, [49, 50, 51]],
    ['a', 3, [52, 53]],
    ['m', 420, [54, 63, 72, 81]],
    ['s', 6, [56, 57]],
    ['a', 20, [58, 66, 67, 76]],
    ['d', 4, [59, 68]],
    ['a', 26, [60, 69, 70, 78]],
    ['d', 3, [61, 62]],
    ['m', 48, [65, 74, 75]],
    ['g', 9, [77]],
    ['m', 162, [71, 79, 80]]
]

iterator(puzzle_207034, 6)


# In[269]:


puzzle_74402 = [
    ['m', 15, [1, 2, 11]],
    ['s', 2, [3, 12]],
    ['s', 3, [4, 13]],
    ['d', 4, [5, 14]],
    ['d', 3, [6, 15]],
    ['m', 6, [7, 16]],
    ['g', 9, [8]],
    ['s', 2, [9, 18]],
    ['m', 24, [10, 19]],
    ['a', 18, [17, 25, 26, 27]],
    ['s', 8, [20, 21]],
    ['m', 336, [22, 23, 24]],
    ['s', 6, [28, 37]],
    ['s', 6, [29, 30]],
    ['a', 6, [31, 32]],
    ['a', 15, [33, 34, 42]],
    ['a', 11, [35, 44, 45]],
    ['g', 6, [36]],
    ['g', 8, [43]],
    ['a', 16, [38, 39, 47]],
    ['s', 1, [40, 41]],
    ['s', 5, [46, 55]],
    ['s', 2, [48, 57]],
    ['s', 3, [49, 58]],
    ['m', 168, [50, 51, 52]],
    ['m', 192, [56, 64, 65, 73]],
    ['m', 45, [53, 61, 62]],
    ['a', 24, [54, 63, 72]],
    ['s', 8, [59, 68]],
    ['d', 2, [60, 69]],
    ['a', 5, [66, 67]],
    ['m', 40, [70, 71, 79]],
    ['d', 2, [80, 81]],
    ['a', 15, [74, 75]],
    ['m', 36, [76, 77]],
    ['g', 5, [78]]
]

iterator(puzzle_74402, 7)


# In[71]:


puzzle_74459 = [
    ['m', 224, [1, 2, 3]],
    ['m', 24, [10, 19, 28]],
    ['a', 23, [5, 6, 14, 15]],
    ['d', 3, [4, 13]],
    ['a', 6, [7, 16, 25]],
    ['s', 1, [8, 9]],
    ['a', 9, [17, 18, 27]],
    ['m', 144, [26, 35, 36]],
    ['s', 3, [11, 12]],
    ['a', 30, [21, 22, 23, 24]],
    ['m', 30, [20, 29, 38, 39]],
    ['m', 224, [30, 31, 40]],
    ['d', 3, [37, 46]],
    ['g', 3, [32]],
    ['m', 210, [33, 34, 43]],
    ['m', 48, [41, 50]],
    ['s', 2, [42, 51]],
    ['a', 19, [44, 45, 54]],
    ['d', 4, [52, 53]],
    ['m', 135, [47, 55, 56]],
    ['s', 1, [48, 57]],
    ['s', 3, [49, 58]],
    ['d', 4, [59, 60]],
    ['m', 504, [61, 62, 63]],
    ['s', 2, [64, 65]],
    ['m', 24, [73, 74]],
    ['s', 3, [66, 75]],
    ['a', 9, [67, 68]],
    ['s', 8, [76, 77]],
    ['d', 2, [69, 78]],
    ['a', 11, [70, 71]],
    ['m', 24, [79, 80]],
    ['a', 10, [72, 81]]
]

iterator(puzzle_74459, 9)


# In[74]:


puzzle_69420 = [
    ['a', 13, [1, 10]],
    ['a', 18, [2, 11, 12]],
    ['g', 2, [3]],
    ['a', 9, [4, 13]],
    ['a', 10, [19, 20, 21, 22]],
    ['a', 22, [5, 6, 14, 23]],
    ['a', 10, [15, 24]],
    ['a', 10, [7, 16]],
    ['a', 6, [8, 9]],
    ['a', 17, [17, 18]],
    ['a', 12, [25, 26]],
    ['a', 19, [28, 29, 37, 46]],
    ['a', 5, [30, 31]],
    ['a', 30, [32, 33, 34, 35]],
    ['a', 9, [27, 36]],
    ['a', 15, [38, 39]],
    ['a', 5, [47, 48]],
    ['a', 9, [40, 49]],
    ['a', 9, [55, 56]],
    ['a', 13, [57, 58]],
    ['a', 30, [64, 65, 66, 67]],
    ['a', 9, [73, 74]],
    ['a', 14, [75, 76]],
    ['a', 8, [41, 42]],
    ['a', 15, [50, 51]],
    ['a', 9, [43, 52]],
    ['a', 8, [44, 45]],
    ['a', 9, [53, 54]],
    ['a', 16, [59, 60, 61]],
    ['a', 12, [62, 63, 72]],
    ['g', 1, [71]],
    ['a', 5, [68, 77]],
    ['a', 12, [69, 70, 78]],
    ['a', 14, [79, 80, 81]]
    
]

iterator(puzzle_69420, 7)


# In[25]:


expert_9 = [
    ['d', 4, [1, 2]],
    ['m', 240, [10, 11, 12]],
    ['a', 15, [19, 20, 21]],
    ['m', 54, [3, 4, 13, 22]],
    ['a', 22, [5, 6, 15]],
    ['g', 2, [14]],
    ['d', 4, [23, 24]],
    ['s', 1, [7, 8]],
    ['a', 9, [16, 25]],
    ['d', 3, [17, 26]],
    ['s', 2, [9, 18]],
    ['m', 56, [27, 36]],
    ['a', 9, [34, 35]],
    ['s', 4, [32, 33]],
    ['d', 2, [30, 31]],
    ['s', 1, [28, 37]],
    ['s', 1, [29, 38]],
    ['m', 24, [46, 47, 55]],
    ['a', 14, [64, 73]],
    ['a', 20, [56, 65, 74]],
    ['a', 18, [39, 40, 49]],
    ['s', 1, [48, 57]],
    ['s', 2, [66, 75]],
    ['s', 4, [67, 76]],
    ['a', 6, [68, 77]],
    ['m', 1176, [41, 50, 58, 59]],
    ['m', 12, [42, 51]],
    ['s', 7, [43, 52]],
    ['m', 24, [44, 53]],
    ['a', 10, [45, 54]],
    ['d', 3, [60, 69]],
    ['g', 1, [61]],
    ['m', 10, [62, 63]],
    ['s', 1, [70, 71]],
    ['a', 11, [78, 79]],
    ['a', 12, [72, 80, 81]]
    
]

iterator(expert_9, 10)


# In[66]:


no_op_9 = [
    [9, [1, 10, 11]],
    [5, [2, 3]],
    [18, [4, 12, 13]],
    [15, [5, 6, 15]],
    [3, [14]],
    [36, [7, 16]],
    [1, [8, 9]],
    [35, [17, 18]],
    [8, [19, 28]],
    [3, [20]],
    [84, [29, 38, 47]],
    [2, [37, 46]],
    [12, [21, 22, 30, 39]],
    [216, [23, 31, 32, 40]],
    [9, [24, 33]],
    [12, [25, 34]],
    [1, [26, 35]],
    [1, [27, 36]],
    [6, [55, 56]],
    [700, [64, 65, 73, 74]],
    [35, [48, 57]],
    [2, [66, 75]],
    [160, [49, 58, 67]],
    [378, [76, 77, 78]],
    [3, [79]],
    [2, [80, 81]],
    [336, [59, 60, 61, 70]],
    [7, [68, 69]],
    [4, [62, 63]],
    [3, [71, 72]],
    [4, [41, 50]],
    [2, [42, 51]],
    [3, [43, 44]],
    [1, [52, 53]],
    [1, [45, 54]]
    
]

iterator(no_op_9, 8)


# In[67]:


chimera_ant = [
    ['a', 21, [1, 2, 3, 10, 19]],
    ['a', 25, [7, 8, 9, 18, 27]],
    ['a', 26, [55, 64, 73, 74, 75]],
    ['a', 25, [63, 72, 79, 80, 81]],
    ['m', 60, [4, 5, 6, 14]],
    ['a', 24, [28, 37, 38, 46]],
    ['a', 17, [36, 44, 45, 54]],
    ['m', 378, [68, 76, 77, 78]],
    ['g', 8, [21]],
    ['g', 2, [23]],
    ['g', 9, [25]],
    ['g', 2, [39]],
    ['g', 3, [43]],
    ['g', 1, [57]],
    ['g', 6, [59]],
    ['g', 7, [61]],
    ['a', 25, [32, 40, 41, 42, 50]],
    ['s', 6, [11, 12]],
    ['a', 13, [16, 17]],
    ['a', 13, [30, 31]],
    ['s', 3, [33, 34]],
    ['a', 13, [48, 49]],
    ['s', 1, [51, 52]],
    ['s', 1, [65, 66]],
    ['m', 3, [70, 71]],
    ['a', 11, [13, 22]],
    ['s', 4, [15, 24]],
    ['a', 8, [20, 29]],
    ['a', 11, [26, 35]],
    ['s', 1, [47, 56]],
    ['a', 11, [53, 62]],
    ['m', 16, [58, 67]],
    ['a', 12, [60, 69]]
]

iterator(chimera_ant, 12)


# In[68]:


miyamoto = [
    [15, [1, 2]],
    [15, [4, 5]],
    [15, [6, 7]],
    [15, [8, 9, 17, 18, 26, 27]],
    [72, [10, 11, 19]],
    [15, [3, 12, 21]],
    [10, [13, 14]],
    [15, [15, 24]],
    [10, [16, 25, 34]],
    [13, [35, 36]],
    [21, [20, 29, 30]],
    [20, [22, 23, 31]],
    [2, [32, 33]],
    [12, [28, 37]],
    [12, [46, 55]],
    [12, [64, 73]],
    [16, [38, 39]],
    [16, [40, 41]],
    [15, [42, 51, 60]],
    [15, [45, 54, 63]],
    [15, [43, 44, 52]],
    [16, [53, 62]],
    [15, [61, 70]],
    [10, [47, 56]],
    [8, [48, 49]],
    [8, [50, 59]],
    [30, [57, 65, 66, 74]],
    [30, [75, 76, 77, 78]],
    [10, [58, 67, 68, 69]],
    [15, [71, 79, 80]],
    [14, [72, 81]]
]

iterator(miyamoto, 8)


# In[180]:


fifteen = [
    ['m', 72, [1, 2]],
    ['a', 23, [16, 17]],
    ['m', 20, [3, 18]],
    ['a', 19, [4, 5]],
    ['d', 15, [6, 7]],
    ['m', 390, [8, 9, 23]],
    ['g', 14, [10]],
    ['a', 11, [11, 12]],
    ['a', 39, [13, 14, 28, 29, 44]],
    ['g', 10, [15]],
    ['g', 4, [30]],
    ['s', 3, [45, 60]],
    ['m', 1200, [19, 33, 34]],
    ['g', 7, [20]],
    ['m', 72, [21, 22]],
    ['m', 16, [24, 25]],
    ['m', 13104, [26, 27, 41, 42, 43]],
    ['g', 7, [31]],
    ['a', 15, [32, 47, 48]],
    ['d', 15, [46, 61]],
    ['s', 5, [35, 50]],
    ['s', 1, [36, 51]],
    ['g', 9, [37]],
    ['m', 52, [38, 53]],
    ['m', 75, [39, 54]],
    ['s', 3, [40, 55]],
    ['g', 8, [62]],
    ['a', 19, [63, 78]],
    ['m', 12, [49, 64]],
    ['m', 132, [65, 66]],
    ['a', 10, [52, 67]],
    ['m', 30, [56, 71]],
    ['a', 27, [57, 72]],
    ['a', 23, [58, 59]],
    ['g', 9, [75]],
    ['g', 15, [105]],
    ['m', 30, [68, 83]],
    ['g', 7, [69]],
    ['m', 90, [70, 85]],
    ['a', 16, [73, 88]],
    ['g', 8, [87]],
    ['m', 56, [74, 89, 90]],
    ['a', 22, [76, 77, 92]],
    ['a', 16, [79, 94]],
    ['s', 9, [80, 95, 110]],
    ['m', 80, [81, 96]],
    ['a', 27, [82, 97]],
    ['s', 6, [84, 99]],
    ['m', 77, [86, 101]],
    ['g', 12, [100]],
    ['a', 19, [91, 106]],
    ['s', 10, [93, 108]],
    ['a', 18, [98, 113]],
    ['a', 4, [102, 103]],
    ['a', 11, [104, 119]],
    ['m', 30, [107, 122]],
    ['m', 56, [109, 124]],
    ['m', 35, [111, 126]],
    ['g', 10, [112]],
    ['a', 18, [114, 129]],
    ['d', 15, [115, 116]],
    ['m', 104, [117, 118]],
    ['m', 77616, [120, 134, 135, 149, 150]],
    ['m', 39, [121, 136]],
    ['a', 6, [151, 166]],
    ['a', 16, [137, 152]],
    ['s', 8, [123, 138]],
    ['s', 4, [125, 140]],
    ['m', 72, [127, 128, 143]],
    ['m', 52, [130, 131]],
    ['m', 108, [132, 146, 147]],
    ['s', 7, [133, 148]],
    ['s', 12, [139, 154]],
    ['g', 1, [153]],
    ['g', 14, [141]],
    ['m', 10, [142, 157]],
    ['m', 440, [144, 145, 160]],
    ['m', 70, [167, 168]],
    ['g', 12, [169]],
    ['d', 13, [170, 185]],
    ['a', 30, [155, 156, 171]],
    ['g', 11, [172]],
    ['g', 14, [174]],
    ['a', 29, [158, 159, 173]],
    ['s', 12, [161, 176]],
    ['a', 20, [162, 177, 192]],
    ['s', 2, [163, 178]],
    ['g', 2, [164]],
    ['s', 1, [165, 180]],
    ['a', 14, [175, 190]],
    ['m', 720, [179, 193, 194, 195, 208]],
    ['g', 15, [209]],
    ['m', 100, [210, 224, 225]],
    ['g', 10, [207]],
    ['m', 91, [222, 223]],
    ['a', 31, [181, 196, 197]],
    ['a', 26, [182, 183, 198]],
    ['a', 16, [184, 199]],
    ['m', 55, [211, 212]],
    ['g', 14, [213]],
    ['g', 3, [214]],
    ['s', 9, [200, 215]],
    ['m', 468, [186, 201, 216]],
    ['m', 32, [202, 217]],
    ['g', 12, [218]],
    ['s', 3, [187, 188, 189, 203]],
    ['s', 4, [191, 206]],
    ['m', 1344, [204, 205, 219, 220, 221]]
]

iterator(fifteen, 4)


# In[11]:


try1 = [2, 4, 5]
try2 = [1]
try3 = [1, 2, 4]
try4 = [8, 10, 12]

try_list = [try1, try2, try3, try4]

print(add_lists_to(try_list, 20))


# In[ ]:




