# my_team.py
# Capture-the-flag Pacman team: belief tracking + cost-aware path planning.

import heapq
import random
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions

CONFIG = dict(
    mode='split',         # split | double | dynamic
    carry_cap=4,          # go home when carrying this much (ghost seen recently)
    cap_unseen=6,         # ... when no ghost was seen recently
    cap_scared=14,        # ... while enemy ghosts are scared
    flee_dist=5,          # ghost closer than this -> run home (if carrying >= flee_carry)
    flee_carry=1,
    lock_lead=0,          # if > 0: everybody defends once we lead by this much
    capsule_bonus=0,      # how attractive capsules are as targets
    split_bias=6,         # attackers prefer different halves of the food
    depth_w=0.0,          # penalty for food far from home
    opportunist=3,        # attacker at home chases invaders this close
    desperate=True,       # defender attacks when losing late in the game
    belief_deadend=False, # use estimated (unseen) ghost positions for dead-end safety
    deadend_margin=1,
    scared_attack=False,  # a scared defender goes attacking
    help_def=True,        # attacker at home helps when there are two invaders
    tempo=0,              # extra cost for cells a ghost reaches before us
    race_guard=False,     # double mode: an empty-handed attacker near home goes back to defend
    late_lock=0,          # everybody defends a lead during the last N moves
    carry_prio=False,     # defenders prefer the invader carrying the most food
    cutoff=False,         # defenders cut invaders off from the border instead of chasing
    win_aware=True,       # go home as soon as the food we carry is enough to end the game
    cap_frac=0.0,         # carry caps are at least this fraction of the food on the map
    stuck_defend=0,       # blocked at the border this many turns -> hunt invaders instead
    vflee=0,              # ghost within this distance while going home -> territory-based escape
    bank_min=0,           # deposit when carrying at least this much ...
    bank_dist=0,          # ... and the border is at most this far away
    bank_ratio=0.0,       # or when carrying >= bank_ratio * distance to the border
    canon='none',         # none | blue | red: break ties as that colour would (mirrored for the other)
    canon_parts='dirs,heap,food,slot,border',
    mm_depth=0,           # minimax look-ahead (own moves) used to veto moves that lose to a nearby ghost
    mm_trig=3,            # ... when a ghost is at most this far away
    est_avoid=0.0,        # extra cost near estimated positions of unseen defenders
    adapt_def=0,          # second agent defends while we trail by at least this much
    cluster_w=0.0,        # prefer food that has other food next to it
    adapt_deaths=0,       # after this many own deaths play cautiously (cz_* values)
    cz_cap=5, cz_flee_carry=2, cz_flee_dist=5,
    safe_plan=0,          # 1: with a ghost in sight only take food that can be reached and brought home before any ghost; 2: else leave
    safe_margin=0, safe_tries=12,
    cap_guard=0,          # >0: second agent waits on our capsule for capsule rushers (invaders carrying at most this much)
    guard_time=3,         # ... until this many moves after the earliest possible arrival of an opponent
    carry_alarm=0,        # an invader carrying at least this much makes both of us hunt it
    scared_min=0,         # double/dynamic: a ghost scared for at least this many more moves attacks instead of defending
    trap=0,               # defender keeps the invader away from the border exits (1: visible invaders, 2: also estimated)
    trap_cap=99,
    trap_caps=0,          # the trapping defender also keeps the invader away from our capsules
    mate_pen=0,           # extra cost for food the attacking teammate is closer to (keeps the two apart)
    leave_gd=0,           # nothing safe to take and a ghost at most this close -> leave by a safe route
    passive_k=0,          # a ghost that declined this many chances to eat us is no longer feared (until it does)
    rooms=0,              # >0: also treat pockets/rooms of at most this many cells behind a single doorway as dead ends
    thr_lock=0.0,         # leading while invaders carry at least this fraction of our lead -> everybody defends
    shallow=-1,           # >=0: ignore invaders that carry nothing and are at most this far inside our border
    bd_carry=0,           # belief_deadend only applies while carrying at least this much
    de_carry=0,           # carrying at least this much -> stay out of dead ends ...
    de_depth=3,           # ... that are at least this deep (while an unscared defender exists)
    esc_commit=0,         # keep heading for an escape capsule this many moves after the ghost is out of sight
    safe_leave=0,         # nothing safe to take while carrying at least this much -> leave (0 = off)
    danger_scale=1.0,     # multiplier of the soft ghost-proximity costs
    ghost_mem=0,          # remember a ghost's last seen position for this many team turns
    stall_home=0,         # carrying food and pacing among <=4 cells for this many turns -> go home
)
CONFIG.update({'mode': 'double', 'split_bias': 0, 'cap_frac': 0.25, 'flee_dist': 5, 'carry_cap': 20, 'flee_carry': 7, 'bank_min': 3, 'bank_dist': 1, 'canon': 'blue', 'depth_w': -0.3, 'ghost_mem': 6, 'stall_home': 12, 'safe_plan': 1, 'trap': 1, 'scared_min': 3, 'esc_commit': 15, 'cap_unseen': 20, 'mate_pen': 20})

DIRS = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}
DIRS_FLIP = {Directions.SOUTH: (0, -1), Directions.NORTH: (0, 1),
             Directions.WEST: (-1, 0), Directions.EAST: (1, 0)}
INF = 999999


def create_team(first_index, second_index, is_red, first='SmartAgent', second='SmartAgent', num_training=0, **kw):
    shared = Shared()
    return [SmartAgent(first_index, shared, 0), SmartAgent(second_index, shared, 1)]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def ipos(p):
    return int(p[0]), int(p[1])


class Shared:
    """Everything the two teammates share: map analysis and enemy beliefs."""

    def __init__(self):
        self.ready = False
        self.roles = {0: 'attack', 1: 'defend'}
        self.targets = {0: None, 1: None}
        self.hists = {0: [], 1: []}

    def setup(self, gs, red, opps):
        self.ready = True
        self.red = red
        self.opps = list(opps)
        walls = gs.get_walls()
        W, H = walls.width, walls.height
        self.W, self.H = W, H
        self.flip = (CONFIG['canon'] == 'blue' and red) or (CONFIG['canon'] == 'red' and not red)
        self.parts = set(CONFIG['canon_parts'].split(',')) if self.flip else set()
        cells = [(x, y) for x in range(W) for y in range(H) if not walls[x][y]]
        cs = set(cells)
        self.cells = cells
        self.nb = {}
        for c in cells:
            lst = []
            for d, (dx, dy) in (DIRS_FLIP if 'dirs' in self.parts else DIRS).items():
                n = (c[0] + dx, c[1] + dy)
                if n in cs:
                    lst.append((n, d))
            self.nb[c] = lst
        # all-pairs maze distances
        self.dist = {}
        for c in cells:
            dd = {c: 0}
            q = deque([c])
            while q:
                u = q.popleft()
                du = dd[u] + 1
                for n, _ in self.nb[u]:
                    if n not in dd:
                        dd[n] = du
                        q.append(n)
            self.dist[c] = dd
        half = W // 2
        self.half = half
        hx = half - 1 if red else half
        ex = half if red else half - 1
        self.border = [(hx, y) for y in range(H) if (hx, y) in cs and (ex, y) in cs]
        self.home_dist = {c: min([self.D(c, b) for b in self.border] or [INF]) for c in cells}
        # dead ends (tree-like parts of the maze)
        deg = {c: len(self.nb[c]) for c in cells}
        pruned = set()
        q = deque([c for c in cells if deg[c] <= 1])
        while q:
            c = q.popleft()
            if c in pruned:
                continue
            pruned.add(c)
            for n, _ in self.nb[c]:
                if n not in pruned:
                    deg[n] -= 1
                    if deg[n] <= 1:
                        q.append(n)
        core = [c for c in cells if c not in pruned]
        self.depth, self.exit = {}, {}
        if core:
            for c in pruned:
                e = min(core, key=lambda k: self.D(c, k))
                if self.D(c, e) < INF:
                    self.exit[c] = e
                    self.depth[c] = self.D(c, e)
        if CONFIG['rooms']:
            # regions cut off from the other half by one single cell (corridors, pockets, rooms)
            for side in (False, True):
                reg = [c for c in cells if self.home(c) == side]
                rs = set(reg)
                entry = [c for c in reg if any(n not in rs for n, _ in self.nb[c])]
                best = {}
                for v in reg:
                    seen = set(e for e in entry if e != v)
                    q = deque(seen)
                    while q:
                        u = q.popleft()
                        for n, _ in self.nb[u]:
                            if n in rs and n != v and n not in seen:
                                seen.add(n)
                                q.append(n)
                    size = len(reg) - len(seen) - 1
                    if 0 < size <= CONFIG['rooms']:
                        for c in reg:
                            if c not in seen and c != v and (c not in best or size > best[c][0]):
                                best[c] = (size, v)
                for c, (size, v) in best.items():
                    if self.D(c, v) < INF:
                        self.exit[c] = v
                        self.depth[c] = self.D(c, v)
        # beliefs
        self.start = {o: ipos(gs.get_initial_agent_position(o)) for o in self.opps}
        self.bel = {o: {self.start[o]: 1.0} for o in self.opps}
        self.prev_def = None
        self.goal = None
        self.food0 = None
        self.ghost_seen = -100
        self.seen = {}
        self.rusher = False
        self.passive = {}
        self.aggr = set()
        self.t_caps = None
        self.ncaps = 0
        self.deaths = 0
        self.turn = 0

    def key(self, c, part='heap'):
        return (self.W - 1 - c[0], self.H - 1 - c[1]) if part in self.parts else c

    def D(self, a, b):
        return self.dist[a].get(b, INF)

    def home(self, c):
        return c[0] < self.half if self.red else c[0] >= self.half

    def best(self, o):
        b = self.bel[o]
        return max(b, key=b.get)

    def update(self, agent, gs):
        self.turn += 1
        team = agent.get_team(gs)
        my_pos = [ipos(gs.get_agent_position(i)) for i in team]
        me = ipos(gs.get_agent_position(agent.index))
        mover = (agent.index - 1) % gs.get_num_agents()
        if mover in self.bel and self.turn > 1:
            nb_ = {}
            for c, p in self.bel[mover].items():
                opts = self.nb[c]
                share = p / (len(opts) + 1)
                nb_[c] = nb_.get(c, 0) + share
                for n, _ in opts:
                    nb_[n] = nb_.get(n, 0) + share
            self.bel[mover] = nb_
        cur = set(agent.get_food_you_are_defending(gs).as_list()) | set(agent.get_capsules_you_are_defending(gs))
        eaten = (self.prev_def - cur) if self.prev_def is not None else set()
        self.prev_def = cur
        noisy = gs.get_agent_distances()
        for o in self.opps:
            st = gs.get_agent_state(o)
            p = st.get_position()
            if p is not None:
                self.bel[o] = {ipos(p): 1.0}
                continue
            if o == mover and eaten:
                self.bel[o] = {f: 1.0 / len(eaten) for f in eaten}
                continue
            nd = noisy[o] if noisy else None

            def ok(c):
                if any(manh(c, q) <= 5 for q in my_pos):
                    return False
                if nd is not None and abs(nd - manh(c, me)) > 6:
                    return False
                return self.home(c) == st.is_pacman

            nb_ = {c: p for c, p in self.bel[o].items() if ok(c)}
            if not nb_:
                s = self.start[o]
                if ok(s):
                    nb_ = {s: 1.0}
                else:
                    nb_ = {c: 1.0 for c in self.cells if ok(c)} or {s: 1.0}
            tot = sum(nb_.values())
            self.bel[o] = {c: p / tot for c, p in nb_.items()}


class SmartAgent(CaptureAgent):

    def __init__(self, index, shared, slot):
        super().__init__(index)
        self.S = shared
        self.slot = slot
        self.hist = []
        self.entry = None
        self.entry_ttl = 0
        self.streak = 0
        self.guard = False
        self.home_ttl = 0
        self.esc = None
        self.esc_ttl = 0
        self.after = {}
        self.pcells = set()

    def register_initial_state(self, gs):
        self.red = gs.is_on_red_team(self.index)
        self.register_team(self.get_team(gs))
        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display
        if not self.S.ready:
            self.S.setup(gs, self.red, self.get_opponents(gs))

    def choose_action(self, gs):
        legal = gs.get_legal_actions(self.index)
        try:
            a = self.decide(gs)
            if a in legal:
                if CONFIG['passive_k']:
                    self.note_after(gs, a)
                return a
        except Exception:
            import os
            if os.environ.get('PACDBG'):
                import traceback
                traceback.print_exc()
        moves = [a for a in legal if a != Directions.STOP]
        return random.choice(moves or legal)

    def note_after(self, gs, a):
        """Remember which ghosts could eat us right after this move."""
        S = self.S
        self.after = {}
        pos = ipos(gs.get_agent_position(self.index))
        dx, dy = DIRS.get(a, (0, 0))
        nxt = (pos[0] + dx, pos[1] + dy)
        if S.home(nxt) or nxt not in S.dist:
            return
        for o in S.opps:
            st = gs.get_agent_state(o)
            p = st.get_position()
            if p is not None and not st.is_pacman and st.scared_timer <= 1:
                d = S.D(nxt, ipos(p))
                if d <= 1:
                    self.after[o] = d

    # ------------------------------------------------------------ planning

    def dijkstra(self, src, block, extra, allowed=None):
        S = self.S
        cost = {src: 0}
        par = {}
        pq = [(0, S.key(src), src)]
        while pq:
            d, _, u = heapq.heappop(pq)
            if d > cost[u]:
                continue
            for n, a in S.nb[u]:
                if n in block or (allowed is not None and not allowed(n)):
                    continue
                nd = d + 1 + extra.get(n, 0)
                if nd < cost.get(n, INF):
                    cost[n] = nd
                    par[n] = (u, a)
                    heapq.heappush(pq, (nd, S.key(n), n))
        return cost, par

    def first_step(self, par, src, tgt):
        if tgt == src:
            return Directions.STOP
        a = None
        c = tgt
        while c != src:
            c, a = par[c]
        return a

    # ------------------------------------------------------------ decision

    def decide(self, gs):
        S, C = self.S, CONFIG
        S.update(self, gs)
        me = gs.get_agent_state(self.index)
        pos = ipos(me.get_position())
        died = bool(self.hist) and manh(self.hist[-1], pos) > 2
        if died:
            S.deaths += 1
        if C['passive_k']:
            for o in self.after:
                if died:
                    S.aggr.add(o)
                else:
                    S.passive[o] = S.passive.get(o, 0) + 1
            self.after = {}
        self.hist.append(pos)
        S.hists[self.slot] = self.hist
        mate_i = [i for i in self.get_team(gs) if i != self.index][0]
        mate = gs.get_agent_state(mate_i)
        mpos = ipos(mate.get_position())

        E = []
        for o in S.opps:
            st = gs.get_agent_state(o)
            p = st.get_position()
            E.append(dict(i=o, pos=ipos(p) if p is not None else None,
                          est=ipos(p) if p is not None else S.best(o),
                          pac=st.is_pacman, scared=st.scared_timer, carry=st.num_carrying))
        ghosts = [e for e in E if not e['pac'] and e['pos'] and e['scared'] <= 3]
        self.pcells = set()
        if C['passive_k']:
            tame = [e for e in ghosts if S.passive.get(e['i'], 0) >= C['passive_k'] and e['i'] not in S.aggr]
            if tame:
                self.pcells = set(e['pos'] for e in tame)
                ghosts = [e for e in ghosts if e not in tame]
        if C['ghost_mem']:
            for e in E:
                if e['pos']:
                    if e in ghosts:
                        S.seen[e['i']] = (e['pos'], S.turn)
                    else:
                        S.seen.pop(e['i'], None)
                elif not e['pac'] and e['scared'] <= 3 and e['i'] in S.seen:
                    p, t = S.seen[e['i']]
                    if S.turn - t <= C['ghost_mem']:
                        ghosts.append(dict(e, pos=p))
                else:
                    S.seen.pop(e['i'], None)
        invaders = [e for e in E if e['pac']]
        if ghosts:
            S.ghost_seen = S.turn

        score = self.get_score(gs)
        moves_left = gs.data.timeleft // 4
        food = self.get_food(gs).as_list()
        alarm = None
        if C['carry_alarm'] and invaders and me.scared_timer <= 2:
            top = max(invaders, key=lambda e: e['carry'])
            if top['carry'] >= C['carry_alarm']:
                alarm = top
        guard = False
        if C['cap_guard']:
            mycaps = list(self.get_capsules_you_are_defending(gs))
            if S.t_caps is None:
                S.t_caps = min([S.D(S.start[o], c) for o in S.opps for c in mycaps] or [0])
                S.ncaps = len(mycaps)
            # an invader that goes for a capsule while carrying (almost) nothing is a capsule rusher
            if len(mycaps) < S.ncaps:
                if any(e['carry'] <= C['cap_guard'] for e in invaders) and S.turn // 2 <= S.t_caps + 15:
                    S.rusher = True
                S.ncaps = len(mycaps)
            for e in invaders:
                if e['pos'] and e['carry'] <= C['cap_guard'] and any(S.D(e['pos'], c) <= 2 for c in mycaps):
                    S.rusher = True
            if mycaps and self.slot == 1 and not me.is_pacman and me.scared_timer == 0:
                if S.rusher:
                    guard = True
                elif S.turn // 2 <= S.t_caps + C['guard_time']:
                    guard = True
                elif any(e['carry'] <= C['cap_guard'] and min(S.D(e['est'], c) for c in mycaps) <= 8 for e in invaders):
                    guard = True
        info = dict(me=me, pos=pos, mate=mate, mpos=mpos, E=E, ghosts=ghosts, invaders=invaders,
                    score=score, moves_left=moves_left, food=food, alarm=alarm, guard=guard)

        role = self.pick_role(gs, info)
        S.roles[self.slot] = role
        if role == 'defend':
            return self.defend(gs, info)
        return self.attack(gs, info, force_home=(role == 'return'))

    def pick_role(self, gs, I):
        S, C = self.S, CONFIG
        me, pos = I['me'], I['pos']
        carrying = me.num_carrying
        if len(I['food']) <= 2:
            return 'return' if me.is_pacman else 'defend'
        if me.is_pacman and I['moves_left'] <= S.home_dist[pos] + 3:
            return 'return'
        if not me.is_pacman and I['moves_left'] <= 8:
            return 'defend'
        if C['lock_lead'] and I['score'] >= C['lock_lead']:
            return 'return' if me.is_pacman else 'defend'
        if C['late_lock'] and I['score'] > 0 and I['moves_left'] <= C['late_lock']:
            return 'return' if me.is_pacman else 'defend'
        inv = I['invaders']
        if C['shallow'] >= 0:
            inv = [e for e in inv if e['carry'] > 0 or S.home_dist.get(e['est'], 0) > C['shallow']]
        mode = C['mode']
        if C['win_aware'] and me.is_pacman and carrying > 0:
            if S.goal is None:
                S.goal = (len(I['food']) + carrying + I['mate'].num_carrying
                          + me.num_returned + I['mate'].num_returned - 2)
            left = S.goal - me.num_returned - I['mate'].num_returned
            mh = S.hists[1 - self.slot]
            mate_stuck = len(mh) >= 10 and len(set(mh[-10:])) <= 3
            if carrying >= left or (carrying + I['mate'].num_carrying >= left and I['mate'].is_pacman
                                    and not mate_stuck):
                return 'return'
        if C['stuck_defend']:
            if me.is_pacman or S.home_dist[pos] > 4:
                self.streak = 0
            else:
                self.streak += 1
            if not inv or me.is_pacman:
                self.guard = False
            elif self.streak >= C['stuck_defend'] and me.scared_timer == 0:
                self.guard = True
            if self.guard:
                return 'defend'
        if C['adapt_def'] and self.slot == 1 and not me.is_pacman and I['score'] <= -C['adapt_def']:
            return 'defend'
        if I['alarm'] is not None:
            return 'return' if me.is_pacman else 'defend'
        if C['thr_lock'] and I['score'] > 0 and me.scared_timer <= 2:
            thr = sum(e['carry'] for e in I['E'] if e['pac'])
            if thr > 0 and thr >= C['thr_lock'] * I['score']:
                return 'return' if me.is_pacman else 'defend'
        if I['guard']:
            return 'defend'
        if mode == 'rush':
            return 'attack'
        if C['scared_min'] and me.scared_timer >= C['scared_min'] and mode != 'split':
            return 'attack'
        if mode == 'split':
            if self.slot == 0:
                if C['help_def'] and not me.is_pacman and len(inv) == 2:
                    return 'defend'
                return 'attack'
            if C['desperate'] and I['score'] < 0 and I['moves_left'] < 80 and not inv:
                return 'attack'
            if C['scared_attack'] and me.scared_timer > 5:
                return 'attack'
            return 'defend'
        if mode == 'double':
            if (C['race_guard'] and me.is_pacman and inv and carrying == 0 and I['mate'].is_pacman
                    and S.home_dist[pos] <= 3 and S.home_dist[pos] <= S.home_dist[I['mpos']]):
                return 'return'
            if me.is_pacman or not inv:
                return 'attack'
            # at home with invaders around: the teammate closer to them defends
            mine = min(S.D(pos, e['est']) for e in inv)
            if I['mate'].is_pacman:
                return 'defend'
            theirs = min(S.D(I['mpos'], e['est']) for e in inv)
            if len(inv) == 2 or mine < theirs or (mine == theirs and self.slot == 1):
                return 'defend'
            return 'attack'
        # dynamic
        need = len(inv)
        if I['score'] > 0:
            need = max(need, 1)
        if need == 0:
            return 'attack'
        if need >= 2:
            return 'return' if (me.is_pacman and carrying == 0 and False) else ('attack' if me.is_pacman else 'defend')

        def dcost(st, p):
            tgt = [e['est'] for e in inv] or S.border
            c = min(S.D(p, t) for t in tgt)
            if st.is_pacman:
                c += 4 + 2 * st.num_carrying
            return c

        mine, theirs = dcost(me, pos), dcost(I['mate'], I['mpos'])
        if mine < theirs or (mine == theirs and self.slot == 1):
            return 'return' if me.is_pacman else 'defend'
        return 'attack'

    # ------------------------------------------------------------ offense

    def danger(self, I):
        """Blocked cells and extra costs caused by enemy ghosts (and invaders if we are scared)."""
        S, C = self.S, CONFIG
        pos, me = I['pos'], I['me']
        block, extra = set(self.pcells), {}
        threats = [(g['pos'], False) for g in I['ghosts']]
        if me.scared_timer > 0:
            threats += [(e['pos'], True) for e in I['invaders'] if e['pos']]
        for gp, at_home in threats:
            for c, d in S.dist[gp].items():
                if d <= 4 and S.home(c) == at_home:
                    if d <= 1:
                        block.add(c)
                    else:
                        extra[c] = extra.get(c, 0) + (40, 10, 3)[d - 2] * C['danger_scale']
                elif C['tempo'] and d <= 7 and not at_home and not S.home(c) and d <= S.D(pos, c):
                    extra[c] = extra.get(c, 0) + C['tempo']
        if C['est_avoid']:
            for e in I['E']:
                if not e['pac'] and not e['pos'] and e['scared'] <= 3:
                    for c, d in S.dist[e['est']].items():
                        if d <= 3 and not S.home(c):
                            extra[c] = extra.get(c, 0) + C['est_avoid'] * (4 - d)
        seal = [g['pos'] for g in I['ghosts']]
        if C['belief_deadend'] and me.num_carrying >= C['bd_carry']:
            seal += [e['est'] for e in I['E'] if not e['pac'] and not e['pos'] and e['scared'] <= 3]
        if seal:
            m = C['deadend_margin']
            for c, dep in S.depth.items():
                if S.home(c):
                    continue
                e = S.exit[c]
                need = S.D(pos, c) + dep + m
                if any(S.D(g, e) <= need for g in seal):
                    block.add(c)
        if C['de_carry'] and me.num_carrying >= C['de_carry']:
            if any(not e['pac'] and e['scared'] <= 5 for e in I['E']):
                my = S.depth.get(pos, 0)
                for c, dep in S.depth.items():
                    if dep >= C['de_depth'] and dep > my and not S.home(c):
                        block.add(c)
        block.discard(pos)
        return block, extra

    def attack(self, gs, I, force_home=False):
        C, S = CONFIG, self.S
        a = self.attack_inner(gs, I, force_home)
        me, pos = I['me'], I['pos']
        if C['mm_depth'] and me.is_pacman and I['ghosts']:
            g = min(I['ghosts'], key=lambda e: S.D(pos, e['pos']))
            if S.D(pos, g['pos']) <= C['mm_trig']:
                caps = set(self.get_capsules(gs))
                memo = {}
                vals = {}
                for n, act in S.nb[pos] + [(pos, Directions.STOP)]:
                    vals[act] = self.mm_after(n, g['pos'], C['mm_depth'], caps, memo)
                if vals.get(a, -INF) <= -500:
                    best = max(vals, key=lambda k: vals[k])
                    if vals[best] > vals.get(a, -INF):
                        return best
        return a

    def mm_after(self, n, g, d, caps, memo):
        """Value of having just moved to n with the ghost at g to move, d own moves left."""
        S = self.S
        if n == g:
            return -1000 - d
        if S.home(n):
            return 1000 + d
        if n in caps:
            return 800 + d
        key = (n, g, d)
        if key in memo:
            return memo[key]
        v = INF
        for gn, _ in S.nb[g] + [(g, None)]:
            if S.home(gn):
                continue
            if gn == n:
                v = -1000 - d
                break
            if d <= 1:
                w = -2 * S.home_dist[n] + min(S.D(gn, n), 5)
            else:
                w = -INF
                for n2, _ in S.nb[n] + [(n, None)]:
                    x = self.mm_after(n2, gn, d - 1, caps, memo)
                    if x > w:
                        w = x
            if w < v:
                v = w
        memo[key] = v
        return v

    def attack_inner(self, gs, I, force_home=False):
        S, C = self.S, CONFIG
        me, pos = I['me'], I['pos']
        carrying = me.num_carrying
        ghosts = I['ghosts']
        gd = min([S.D(pos, g['pos']) for g in ghosts] or [99])
        defenders = [e for e in I['E'] if not e['pac']]
        all_scared = bool(defenders) and all(e['scared'] > 5 for e in defenders)

        # opportunistic kill while still at home
        if not me.is_pacman and me.scared_timer == 0 and not force_home:
            near = [e for e in I['invaders'] if e['pos'] and S.D(pos, e['pos']) <= C['opportunist']
                    and (C['shallow'] < 0 or e['carry'] > 0 or S.home_dist[e['pos']] > C['shallow']
                         or S.D(pos, e['pos']) <= 1)]
            if near:
                return self.defend(gs, I)

        if S.food0 is None:
            S.food0 = len(I['food']) + carrying + I['mate'].num_carrying
        cautious = C['adapt_deaths'] and S.deaths >= C['adapt_deaths']
        flee_dist = C['cz_flee_dist'] if cautious else C['flee_dist']
        flee_carry = C['cz_flee_carry'] if cautious else C['flee_carry']
        if all_scared:
            cap = C['cap_scared']
        elif cautious:
            cap = C['cz_cap']
        elif S.turn - S.ghost_seen < 30:
            cap = C['carry_cap']
        else:
            cap = C['cap_unseen']
        if not cautious or all_scared:
            cap = max(cap, int(C['cap_frac'] * S.food0))
        go_home = force_home or (carrying > 0 and (
            carrying >= cap or (me.is_pacman and gd <= flee_dist and carrying >= flee_carry)))
        hd = S.home_dist[pos]
        if me.is_pacman and carrying > 0:
            if C['bank_min'] and carrying >= C['bank_min'] and hd <= C['bank_dist']:
                go_home = True
            if C['bank_ratio'] and carrying >= max(2, C['bank_ratio'] * hd):
                go_home = True
        if C['stall_home'] and me.is_pacman and carrying > 0:
            n = C['stall_home']
            if self.home_ttl > 0:
                self.home_ttl -= 1
                go_home = True
            elif len(self.hist) >= n and len(set(self.hist[-n:])) <= 4:
                self.home_ttl = 25
                go_home = True
        else:
            self.home_ttl = 0
        if all_scared and carrying > 0:
            t = min(e['scared'] for e in defenders)
            if t <= S.home_dist[pos] + 2:
                go_home = True

        block, extra = self.danger(I)
        cost, par = self.dijkstra(pos, block, extra)

        if C['safe_plan'] and ghosts and not all_scared:
            a = self.safe_move(gs, I, go_home, gd)
            if a is not None:
                return a

        # stuck in front of a guarded border: try another entrance
        if not me.is_pacman and not go_home:
            if self.entry_ttl > 0:
                self.entry_ttl -= 1
                if pos == self.entry:
                    self.entry_ttl = 0
            elif ghosts and len(self.hist) >= 12 and len(set(self.hist[-12:])) <= 4:
                ranked = sorted(S.border, key=lambda b: -min(S.D(b, g['pos']) for g in ghosts))
                self.entry = random.choice(ranked[:3])
                self.entry_ttl = 20
            if self.entry_ttl > 0 and self.entry in cost:
                hc, hp = self.dijkstra(pos, set(), {}, S.home)
                if self.entry in hc:
                    return self.first_step(hp, pos, self.entry)

        capsules = [c for c in self.get_capsules(gs) if c in cost]
        if go_home:
            if not me.is_pacman:
                return self.defend(gs, I)
            if self.esc_ttl > 0 and not all_scared:
                self.esc_ttl -= 1
                if self.esc in capsules and self.esc != pos:
                    return self.first_step(par, pos, self.esc)
                self.esc_ttl = 0
            if C['vflee'] and ghosts and gd <= C['vflee'] and not all_scared:
                a = self.vflee(gs, I)
                if a is not None:
                    return a
            tg = [(cost[b], S.key(b, 'border'), b) for b in S.border if b in cost]
            if ghosts and not all_scared:
                tg += [(cost[c] - 2, S.key(c, 'border'), c) for c in capsules]
            if tg:
                return self.first_step(par, pos, min(tg)[-1])
            return self.flee(gs, I)

        cand = self.rank(I, cost, capsules, gd, all_scared)
        if cand:
            tgt = min(cand)[-1]
            S.targets[self.slot] = tgt
            return self.first_step(par, pos, tgt)
        S.targets[self.slot] = None
        if me.is_pacman:
            tg = [(cost[b], S.key(b, 'border'), b) for b in S.border if b in cost]
            if tg:
                return self.first_step(par, pos, min(tg)[-1])
            return self.flee(gs, I)
        return self.defend(gs, I)

    def rank(self, I, cost, capsules, gd, all_scared):
        """Candidate targets (score, tie-break, cell); lower score is better."""
        S, C = self.S, CONFIG
        me = I['me']
        mate_attacks = S.roles.get(1 - self.slot) == 'attack'
        mt = S.targets.get(1 - self.slot)
        ys = sorted(S.key(f, 'slot')[1] for f in I['food'])
        med = ys[len(ys) // 2] if ys else 0
        cand = []
        foodset = set(I['food'])
        for f in I['food']:
            if f not in cost:
                continue
            s = cost[f] + C['depth_w'] * S.home_dist[f]
            if C['cluster_w']:
                s -= C['cluster_w'] * sum(1 for n, _ in S.nb[f] if n in foodset)
            if mate_attacks:
                upper = S.key(f, 'slot')[1] >= med
                if upper != (self.slot == 0):
                    s += C['split_bias']
                if mt and manh(f, mt) <= 2:
                    s += 4
                if C['mate_pen']:
                    dm = S.D(I['mpos'], f)
                    if dm < cost[f] or (dm == cost[f] and self.slot == 1):
                        s += C['mate_pen']
            cand.append((s, S.key(f, 'food'), f))
        if not all_scared:
            for c in capsules:
                s = cost[c] - C['capsule_bonus']
                if me.is_pacman and gd <= 6:
                    s -= 10
                cand.append((s, S.key(c, 'food'), c))
        return cand

    def reach(self, src, t0, Dg, m, caps=(), want_exit=False):
        """Cells reachable from src (leaving at time t0) strictly before any ghost can get there.
        want_exit: only answer whether home (or a capsule) can be reached that way."""
        S = self.S
        arr = {src: t0}
        par = {}
        q = deque([src])
        while q:
            u = q.popleft()
            k = arr[u] + 1
            for n, a in S.nb[u]:
                if n in arr or n in self.pcells:
                    continue
                if S.home(n):
                    if want_exit:
                        return True
                else:
                    lim = k + 1 + m
                    bad = False
                    for dg in Dg:
                        if dg.get(n, INF) < lim:
                            bad = True
                            break
                    if bad:
                        continue
                    if want_exit and n in caps:
                        return True
                arr[n] = k
                par[n] = (u, a)
                q.append(n)
        return False if want_exit else (arr, par)

    def safe_move(self, gs, I, go_home, gd):
        S, C = self.S, CONFIG
        me, pos = I['me'], I['pos']
        Dg = [S.dist[g['pos']] for g in I['ghosts']]
        m = C['safe_margin']
        caps = set(self.get_capsules(gs))
        arr, par = self.reach(pos, 0, Dg, m)

        def leave():
            tg = [(arr[b], S.key(b, 'border'), b) for b in S.border if b in arr]
            tg += [(arr[c] - 2, S.key(c, 'border'), c) for c in caps if c in arr]
            if tg:
                t = min(tg)[-1]
                if t in caps:
                    self.esc, self.esc_ttl = t, C['esc_commit']
                else:
                    self.esc_ttl = 0
                return self.first_step(par, pos, t)
            return None

        if go_home:
            return leave() if me.is_pacman else None
        cand = self.rank(I, arr, [c for c in caps if c in arr], gd, False)
        cand.sort()
        tries = 0
        for s, _, f in cand:
            if f in caps:
                ok = True
            else:
                tries += 1
                if tries > C['safe_tries']:
                    break
                ok = self.reach(f, arr[f], Dg, m, caps, True)
            if ok:
                S.targets[self.slot] = f
                return self.first_step(par, pos, f)
        if me.is_pacman and (C['safe_plan'] >= 2 or (C['safe_leave'] and me.num_carrying >= C['safe_leave'])
                             or (C['leave_gd'] and gd <= C['leave_gd'])):
            return leave()
        return None

    def vflee(self, gs, I):
        """Escape by territory: prefer moves from which home (or a capsule) is reachable
        strictly before any ghost, otherwise the move keeping the largest safe area."""
        S = self.S
        pos = I['pos']
        gps = [g['pos'] for g in I['ghosts']]
        caps = set(self.get_capsules(gs))
        best, ba = None, None
        for n, a in S.nb[pos]:
            if any(S.D(g, n) <= 1 for g in gps):
                continue
            seen = {n}
            q = deque([n])
            dn = S.dist[n]
            while q:
                u = q.popleft()
                for v, _ in S.nb[u]:
                    if v in seen:
                        continue
                    dv = dn[v] + 1
                    if any(S.D(g, v) <= dv for g in gps):
                        continue
                    seen.add(v)
                    q.append(v)
            homes = [c for c in seen if S.home(c)]
            if homes:
                key = (3, -min(dn[c] for c in homes), 0)
            else:
                cs = [c for c in seen if c in caps]
                if cs:
                    key = (2, -min(dn[c] for c in cs), 0)
                else:
                    key = (1, min(len(seen), 60), min(S.D(g, n) for g in gps))
            if best is None or key > best:
                best, ba = key, a
        return ba

    def flee(self, gs, I):
        S = self.S
        pos = I['pos']
        gp = [g['pos'] for g in I['ghosts']]
        best, ba = None, Directions.STOP
        for n, a in S.nb[pos]:
            v = (min([S.D(n, g) for g in gp] or [99]), -S.depth.get(n, 0), -S.home_dist[n])
            if best is None or v > best:
                best, ba = v, a
        return ba

    # ------------------------------------------------------------ defense

    def patrol_point(self, e):
        S = self.S
        return min(S.border, key=lambda b: (S.D(b, e['est']), abs(b[1] - S.H / 2)))

    def defend(self, gs, I):
        S, C = self.S, CONFIG
        me, pos = I['me'], I['pos']
        E, inv = I['E'], I['invaders']

        def tp(e):
            return e['est'] if e['pac'] else self.patrol_point(e)

        mate_def = S.roles.get(1 - self.slot) == 'defend' and not I['mate'].is_pacman
        if mate_def and len(E) == 2:
            a = S.D(pos, tp(E[0])) + S.D(I['mpos'], tp(E[1]))
            b = S.D(pos, tp(E[1])) + S.D(I['mpos'], tp(E[0]))
            if inv and len(inv) == 1:
                # closest of us takes the invader
                e_inv = inv[0]
                other = [e for e in E if e is not e_inv][0]
                mine, theirs = S.D(pos, tp(e_inv)), S.D(I['mpos'], tp(e_inv))
                tgt_e = e_inv if (mine < theirs or (mine == theirs and self.slot == 1)) else other
            else:
                tgt_e = E[0] if (a < b or (a == b and self.slot == 0)) else E[1]
        elif inv:
            if C['carry_prio']:
                tgt_e = min(inv, key=lambda e: (-e['carry'], S.D(pos, e['est'])))
            else:
                tgt_e = min(inv, key=lambda e: S.D(pos, e['est']))
        else:
            tgt_e = min(E, key=lambda e: S.home_dist.get(e['est'], INF))
        if I['alarm'] is not None:
            tgt_e = I['alarm']
        tgt = tp(tgt_e)
        gcaps = list(self.get_capsules_you_are_defending(gs)) if I['guard'] else []
        if gcaps and not tgt_e['pac']:
            # nobody inside yet: wait on the capsule closest to an opponent
            tgt = min(gcaps, key=lambda c: (min(S.D(e['est'], c) for e in E), c))

        block = set()
        if me.scared_timer > 0:
            for e in inv:
                if e['pos']:
                    block |= {c for c, d in S.dist[e['pos']].items() if d <= 1}
            if tgt_e['pac'] and tgt_e['pos']:
                ring = [c for c, d in S.dist[tgt_e['pos']].items() if d == 2 and S.home(c)]
                if pos in block:
                    best, ba = None, Directions.STOP
                    for n, a in S.nb[pos]:
                        if S.home(n):
                            v = S.D(n, tgt_e['pos'])
                            if best is None or v > best:
                                best, ba = v, a
                    return ba
                if ring:
                    tgt = min(ring, key=lambda c: S.D(pos, c))
            block.discard(pos)
        if ((C['trap'] or gcaps) and tgt_e['pac'] and me.scared_timer == 0 and S.home(pos) and S.home(tgt_e['est'])
                and (tgt_e['pos'] or C['trap'] >= 2 or I['alarm'] is not None or gcaps)):
            ip = tgt_e['est']
            mp = I['mpos'] if (mate_def and I['mate'].scared_timer == 0) else None
            hc, _ = self.dijkstra(ip, set(), {}, S.home)
            if gcaps:
                exits = gcaps
            else:
                exits = list(S.border)
                if C['trap_caps']:
                    exits += list(self.get_capsules_you_are_defending(gs))
            best, ba = None, None
            for n, a in S.nb[pos] + [(pos, Directions.STOP)]:
                if not S.home(n):
                    continue
                if n == ip and tgt_e['pos']:
                    return a
                f = INF
                dn = S.dist[n]
                di = S.dist[ip]
                for b in exits:
                    dpb = di.get(b, INF)
                    if dpb < f and dn.get(b, INF) > dpb and (mp is None or S.D(mp, b) > dpb):
                        f = dpb
                key = (min(f, C['trap_cap']), -hc.get(n, INF), a != Directions.STOP)
                if best is None or key > best:
                    best, ba = key, a
            if ba is not None:
                return ba
        if C['cutoff'] and tgt_e['pac'] and me.scared_timer == 0 and S.home(pos):
            ip = tgt_e['est']
            if tgt_e['pos'] and S.D(pos, ip) <= 1:
                for n, a in S.nb[pos]:
                    if n == ip:
                        return a
            best, ba = None, Directions.STOP
            for n, a in S.nb[pos] + [(pos, Directions.STOP)]:
                if not S.home(n):
                    continue
                sl = min(S.D(ip, b) - S.D(n, b) for b in S.border)
                v = (min(sl, 1), -S.D(n, ip), a != Directions.STOP)
                if best is None or v > best:
                    best, ba = v, a
            return ba
        cost, par = self.dijkstra(pos, block, {}, S.home)
        if tgt not in cost:
            reach = [c for c in cost if c != pos]
            if not reach:
                return Directions.STOP
            tgt = min(reach, key=lambda c: S.D(c, tgt))
        return self.first_step(par, pos, tgt)
