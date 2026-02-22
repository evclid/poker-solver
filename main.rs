// poker_solver.rs — DCFR Poker Solver
// cargo build --release && ./target/release/poker_solver

use rand::Rng;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  CARDS
// ═══════════════════════════════════════════════════════════════════

const RANK_CHARS: &[u8; 13] = b"23456789TJQKA";
const SUIT_SYMS: [&str; 4] = ["♠", "♥", "♦", "♣"];
const PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

#[inline(always)]
const fn rank_of(c: u8) -> u8 { c >> 2 }
#[inline(always)]
const fn suit_of(c: u8) -> u8 { c & 3 }

fn card_str(c: u8) -> String {
    format!("{}{}", RANK_CHARS[rank_of(c) as usize] as char, SUIT_SYMS[suit_of(c) as usize])
}

fn cards_str(cs: &[u8]) -> String {
    cs.iter().map(|&c| card_str(c)).collect::<Vec<_>>().join(" ")
}

fn shuffle_deck(deck: &mut [u8; 52], rng: &mut impl Rng) {
    for i in (1..52).rev() { let j = rng.gen_range(0..=i); deck.swap(i, j); }
}

fn fresh_deck() -> [u8; 52] {
    let mut d = [0u8; 52];
    (0..52).for_each(|i| d[i] = i as u8);
    d
}

// ═══════════════════════════════════════════════════════════════════
//  HAND EVALUATOR — Cactus Kev style, lookup tables
// ═══════════════════════════════════════════════════════════════════

struct EvalTables {
    flush_lookup: Vec<u16>,    // 8192 entries, bit_pattern -> rank
    unique5_lookup: Vec<u16>,  // 8192 entries
    product_lookup: FxHashMap<u32, u16>,
}

impl EvalTables {
    fn new() -> Self {
        let mut flush_lookup = vec![0u16; 8192];
        let mut unique5_lookup = vec![0u16; 8192];
        let mut product_lookup = FxHashMap::default();

        let (mut straights, mut non_straights) = (Vec::new(), Vec::new());
        for bits in 0u16..8192 {
            if bits.count_ones() != 5 { continue; }
            if Self::is_straight(bits) { straights.push(bits); }
            else { non_straights.push(bits); }
        }

        straights.sort_by_key(|&b| std::cmp::Reverse(Self::straight_high(b)));
        non_straights.sort_by(|a, b| Self::rank_key(*b).cmp(&Self::rank_key(*a)));

        let mut rc: u16 = 1;

        // Straight flushes
        for &b in &straights { flush_lookup[b as usize] = rc; rc += 1; }

        // Four of a kind, Full house
        for (gen_fn, _) in &[
            (Self::gen_foak as fn() -> Vec<u32>, "4oak"),
            (Self::gen_full_house as fn() -> Vec<u32>, "fh"),
        ] {
            for product in gen_fn() { product_lookup.insert(product, rc); rc += 1; }
        }

        // Flush (non-straight)
        for &b in &non_straights { flush_lookup[b as usize] = rc; rc += 1; }
        // Straight (non-flush)
        for &b in &straights { unique5_lookup[b as usize] = rc; rc += 1; }

        // Three of a kind, Two pair, One pair
        for gen_fn in &[
            Self::gen_three_kind as fn() -> Vec<u32>,
            Self::gen_two_pair,
            Self::gen_one_pair,
        ] {
            for product in gen_fn() { product_lookup.insert(product, rc); rc += 1; }
        }

        // High card
        for &b in &non_straights { unique5_lookup[b as usize] = rc; rc += 1; }

        Self { flush_lookup, unique5_lookup, product_lookup }
    }

    #[inline]
    fn is_straight(bits: u16) -> bool {
        bits == 0b1_0000_0000_1111 || (bits >> bits.trailing_zeros()) == 0b11111
    }

    #[inline]
    fn straight_high(bits: u16) -> u8 {
        if bits == 0b1_0000_0000_1111 { 3 } else { 15 - bits.leading_zeros() as u8 }
    }

    fn rank_key(bits: u16) -> u64 {
        let mut key = 0u64;
        for r in (0..13u8).rev() {
            if bits & (1 << r) != 0 { key = key * 16 + r as u64; }
        }
        key
    }

    // Generators return Vec<prime_product> sorted by strength (strongest first)
    fn gen_foak() -> Vec<u32> {
        let mut v = Vec::with_capacity(156);
        for q in (0..13).rev() {
            for k in (0..13).rev() {
                if k != q { v.push(PRIMES[q].pow(4) * PRIMES[k]); }
            }
        }
        v
    }

    fn gen_full_house() -> Vec<u32> {
        let mut v = Vec::with_capacity(156);
        for t in (0..13).rev() {
            for p in (0..13).rev() {
                if p != t { v.push(PRIMES[t].pow(3) * PRIMES[p].pow(2)); }
            }
        }
        v
    }

    fn gen_three_kind() -> Vec<u32> {
        let mut v = Vec::with_capacity(858);
        for t in (0..13usize).rev() {
            for k1 in (0..13usize).rev() {
                if k1 == t { continue; }
                for k2 in (0..k1).rev() {
                    if k2 != t { v.push(PRIMES[t].pow(3) * PRIMES[k1] * PRIMES[k2]); }
                }
            }
        }
        v
    }

    fn gen_two_pair() -> Vec<u32> {
        let mut v = Vec::with_capacity(858);
        for p1 in (0..13usize).rev() {
            for p2 in (0..p1).rev() {
                for k in (0..13usize).rev() {
                    if k != p1 && k != p2 {
                        v.push(PRIMES[p1].pow(2) * PRIMES[p2].pow(2) * PRIMES[k]);
                    }
                }
            }
        }
        v
    }

    fn gen_one_pair() -> Vec<u32> {
        let mut v = Vec::with_capacity(2860);
        for p in (0..13usize).rev() {
            for k1 in (0..13usize).rev() {
                if k1 == p { continue; }
                for k2 in (0..k1).rev() {
                    if k2 == p { continue; }
                    for k3 in (0..k2).rev() {
                        if k3 != p {
                            v.push(PRIMES[p].pow(2) * PRIMES[k1] * PRIMES[k2] * PRIMES[k3]);
                        }
                    }
                }
            }
        }
        v
    }

    #[inline]
    fn eval5(&self, c: &[u8; 5]) -> u16 {
        let mut bit_or = 0u16;
        let mut pp = 1u32;
        let s0 = suit_of(c[0]);
        let mut all_same = true;

        for i in 0..5 {
            let r = rank_of(c[i]);
            bit_or |= 1 << r;
            pp *= PRIMES[r as usize];
            if suit_of(c[i]) != s0 { all_same = false; }
        }

        if all_same { return self.flush_lookup[bit_or as usize]; }
        if bit_or.count_ones() == 5 {
            let v = self.unique5_lookup[bit_or as usize];
            if v != 0 { return v; }
        }
        self.product_lookup.get(&pp).copied().unwrap_or(7462)
    }

    #[inline]
    fn eval7(&self, cards: &[u8; 7]) -> u16 {
        const C: [[u8; 5]; 21] = [
            [0,1,2,3,4],[0,1,2,3,5],[0,1,2,3,6],[0,1,2,4,5],[0,1,2,4,6],
            [0,1,2,5,6],[0,1,3,4,5],[0,1,3,4,6],[0,1,3,5,6],[0,1,4,5,6],
            [0,2,3,4,5],[0,2,3,4,6],[0,2,3,5,6],[0,2,4,5,6],[0,3,4,5,6],
            [1,2,3,4,5],[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,3,4,5,6],
            [2,3,4,5,6],
        ];
        let mut best = u16::MAX;
        for idx in &C {
            let h = [cards[idx[0] as usize], cards[idx[1] as usize],
                      cards[idx[2] as usize], cards[idx[3] as usize], cards[idx[4] as usize]];
            let v = self.eval5(&h);
            if v < best { best = v; }
        }
        best
    }
}

fn rank_name(rank: u16) -> &'static str {
    match rank {
        1..=10 => "Straight Flush",
        11..=166 => "Four of a Kind",
        167..=322 => "Full House",
        323..=1599 => "Flush",
        1600..=1609 => "Straight",
        1610..=2467 => "Three of a Kind",
        2468..=3325 => "Two Pair",
        3326..=6185 => "One Pair",
        6186..=7462 => "High Card",
        _ => "?",
    }
}

// ═══════════════════════════════════════════════════════════════════
//  HAND ABSTRACTIONS
// ═══════════════════════════════════════════════════════════════════

const N_EQ_BUCKETS: usize = 20;
const NUM_ACTIONS: usize = 7;
const BET_FRACTIONS: [f32; NUM_ACTIONS] = [0.0, 0.0, 0.33, 0.50, 0.75, 1.0, 1.5];

fn preflop_canonical(h0: u8, h1: u8) -> u16 {
    let (r0, r1) = (rank_of(h0), rank_of(h1));
    let (hi, lo) = (r0.max(r1), r0.min(r1));
    let suited = suit_of(h0) == suit_of(h1);
    if suited || hi == lo { (hi as u16) * 13 + lo as u16 }
    else { (lo as u16) * 13 + hi as u16 + 169 }
}

fn precompute_preflop_buckets(et: &EvalTables, n_buckets: usize, rng: &mut impl Rng) -> Vec<u8> {
    let mut equities = vec![0.0f32; 338]; // 169*2

    for hi in 0..13u8 {
        for lo in 0..=hi {
            for &suited in &[true, false] {
                if suited && hi == lo { continue; }
                let idx = if suited { (hi as usize) * 13 + lo as usize }
                          else if hi == lo { (hi as usize) * 13 + hi as usize }
                          else { (lo as usize) * 13 + hi as usize + 169 };

                let c0 = hi * 4;
                let c1 = if suited { lo * 4 } else { lo * 4 + 1 };
                let mut dead = [false; 52];
                dead[c0 as usize] = true;
                dead[c1 as usize] = true;
                let live: Vec<u8> = (0..52u8).filter(|&c| !dead[c as usize]).collect();

                let (mut wins, mut total) = (0u32, 0u32);
                let mut buf = live.clone();
                for _ in 0..2000 {
                    for i in 0..7 { let j = rng.gen_range(i..buf.len()); buf.swap(i, j); }
                    let mut m7 = [c0, c1, buf[0], buf[1], buf[2], buf[3], buf[4]];
                    let my_r = et.eval7(&m7);
                    m7[0] = buf[5]; m7[1] = buf[6];
                    let op_r = et.eval7(&m7);
                    total += 1;
                    if my_r < op_r { wins += 2; } else if my_r == op_r { wins += 1; }
                }
                if idx < equities.len() { equities[idx] = wins as f32 / (total as f32 * 2.0); }
            }
        }
    }

    let mut indexed: Vec<_> = equities.iter().enumerate()
        .filter(|(_, &e)| e > 0.0).map(|(i, &e)| (i, e)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut buckets = vec![0u8; equities.len()];
    let per = indexed.len().max(1) / n_buckets.max(1);
    for (pos, &(idx, _)) in indexed.iter().enumerate() {
        buckets[idx] = ((pos / per.max(1)) as u8).min(n_buckets as u8 - 1);
    }
    buckets
}

// ═══════════════════════════════════════════════════════════════════
//  EQUITY CALCULATION
// ═══════════════════════════════════════════════════════════════════

fn mc_equity(et: &EvalTables, hole: [u8; 2], board: &[u8], n_sim: u32, rng: &mut impl Rng) -> f32 {
    let (n_board, remain) = (board.len(), 5 - board.len());
    let mut dead = [false; 52];
    dead[hole[0] as usize] = true;
    dead[hole[1] as usize] = true;
    for &b in board { dead[b as usize] = true; }
    let live: Vec<u8> = (0..52u8).filter(|&c| !dead[c as usize]).collect();
    let (n_live, need) = (live.len(), 2 + remain);

    let (mut wins, mut ties) = (0u32, 0u32);
    let mut buf = live.clone();

    for _ in 0..n_sim {
        for i in 0..need { let j = rng.gen_range(i..n_live); buf.swap(i, j); }

        let mut my7 = [0u8; 7];
        let mut opp7 = [0u8; 7];
        my7[0] = hole[0]; my7[1] = hole[1];
        opp7[0] = buf[0]; opp7[1] = buf[1];
        for i in 0..n_board { my7[2+i] = board[i]; opp7[2+i] = board[i]; }
        for i in 0..remain { my7[2+n_board+i] = buf[2+i]; opp7[2+n_board+i] = buf[2+i]; }

        let (mr, or) = (et.eval7(&my7), et.eval7(&opp7));
        if mr < or { wins += 1; } else if mr == or { ties += 1; }
    }
    (wins as f32 + ties as f32 * 0.5) / n_sim as f32
}

fn postflop_equity(et: &EvalTables, hole: [u8; 2], board: &[u8], rng: &mut impl Rng) -> f32 {
    if board.len() < 3 { return mc_equity(et, hole, board, 500, rng); }

    let mut dead = [false; 52];
    dead[hole[0] as usize] = true;
    dead[hole[1] as usize] = true;
    for &b in board { dead[b as usize] = true; }

    if board.len() == 5 {
        // River: exact enumeration C(45,2)=990
        let mut my7 = [0u8; 7];
        my7[0] = hole[0]; my7[1] = hole[1];
        my7[2..7].copy_from_slice(board);
        let my_rank = et.eval7(&my7);
        let (mut wins, mut ties, mut total) = (0u32, 0u32, 0u32);

        for o0 in 0..51u8 {
            if dead[o0 as usize] { continue; }
            for o1 in (o0+1)..52u8 {
                if dead[o1 as usize] { continue; }
                let o7 = [o0, o1, board[0], board[1], board[2], board[3], board[4]];
                let or = et.eval7(&o7);
                total += 1;
                if my_rank < or { wins += 1; } else if my_rank == or { ties += 1; }
            }
        }
        return (wins as f32 + ties as f32 * 0.5) / total as f32;
    }

    mc_equity(et, hole, board, 1000, rng)
}

// ═══════════════════════════════════════════════════════════════════
//  INFOSET KEY — inline FNV-1a without allocation
// ═══════════════════════════════════════════════════════════════════

#[inline]
fn make_infoset_key(street: u8, hand_bucket: u8, spr_b: u8, history: &[u8]) -> u64 {
    const P: u64 = 1099511628211;
    let mut h: u64 = 14695981039346656037;
    for &b in &[street, hand_bucket, spr_b] {
        h ^= b as u64; h = h.wrapping_mul(P);
    }
    for &b in history { h ^= b as u64; h = h.wrapping_mul(P); }
    h
}

// ═══════════════════════════════════════════════════════════════════
//  DCFR NODE & SOLVER
// ═══════════════════════════════════════════════════════════════════

#[derive(Serialize, Deserialize, Clone)]
struct CfrNode {
    regret_sum: [f32; NUM_ACTIONS],
    strategy_sum: [f32; NUM_ACTIONS],
}

impl CfrNode {
    #[inline]
    fn new() -> Self { Self { regret_sum: [0.0; NUM_ACTIONS], strategy_sum: [0.0; NUM_ACTIONS] } }

    #[inline]
    fn get_strategy(&self) -> [f32; NUM_ACTIONS] {
        let mut s = [0.0f32; NUM_ACTIONS];
        let mut total = 0.0f32;
        for i in 0..NUM_ACTIONS { s[i] = self.regret_sum[i].max(0.0); total += s[i]; }
        if total > 0.0 { for v in &mut s { *v /= total; } }
        else { s = [1.0 / NUM_ACTIONS as f32; NUM_ACTIONS]; }
        s
    }

    #[inline]
    fn avg_strategy(&self) -> [f32; NUM_ACTIONS] {
        let total: f32 = self.strategy_sum.iter().sum();
        if total > 0.0 {
            let mut s = [0.0; NUM_ACTIONS];
            for i in 0..NUM_ACTIONS { s[i] = self.strategy_sum[i] / total; }
            s
        } else { [1.0 / NUM_ACTIONS as f32; NUM_ACTIONS] }
    }
}

#[inline]
fn spr_bucket(pot: i32, stack: i32) -> u8 {
    if stack <= 0 || pot <= 0 { return 0; }
    let spr = stack as f32 / pot as f32;
    const THRESHOLDS: [f32; 9] = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 30.0];
    THRESHOLDS.iter().position(|&t| spr < t).unwrap_or(9) as u8
}

#[inline]
fn equity_bucket(eq: f32) -> u8 { ((eq * N_EQ_BUCKETS as f32) as u8).min(N_EQ_BUCKETS as u8 - 1) }

#[inline]
fn sample_action(strat: &[f32; NUM_ACTIONS], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in strat.iter().enumerate() { cum += p; if r < cum { return i; } }
    NUM_ACTIONS - 1
}

fn board_for_street(b: &[u8; 5], s: u8) -> &[u8] {
    match s { 0 => &[], 1 => &b[..3], 2 => &b[..4], _ => &b[..5] }
}

#[derive(Serialize, Deserialize)]
struct Solver {
    nodes: FxHashMap<u64, CfrNode>,
    preflop_buckets: Vec<u8>,
    iteration: u32,
}

impl Solver {
    fn new(et: &EvalTables) -> Self {
        let mut rng = rand::thread_rng();
        eprintln!("  Precomputing preflop abstractions...");
        let pb = precompute_preflop_buckets(et, N_EQ_BUCKETS, &mut rng);
        eprintln!("  Done.");
        Self { nodes: FxHashMap::default(), preflop_buckets: pb, iteration: 0 }
    }

    #[inline]
    fn hand_bucket(&self, et: &EvalTables, hole: [u8; 2], board: &[u8], street: u8, rng: &mut impl Rng) -> u8 {
        if street == 0 {
            let c = preflop_canonical(hole[0], hole[1]) as usize;
            return if c < self.preflop_buckets.len() { self.preflop_buckets[c] }
                   else { (N_EQ_BUCKETS / 2) as u8 };
        }
        // Используем 50 симуляций для молниеносного обхода дерева
        let eq = mc_equity(et, hole, board, 50, rng);
        equity_bucket(eq)
    }

    fn is_street_done(history: &[u8]) -> bool {
        let n = history.len();
        if n < 2 { return false; }
        let (last, prev) = (history[n-1], history[n-2]);
        // check-check, or call after raise
        if last == 1 && (prev == 1 && n == 2 || prev >= 2) { return true; }
        // Cap at 3 raises
        if last == 1 && history.iter().filter(|&&a| a >= 2).count() >= 3 { return true; }
        // check-bet-call
        if n >= 3 && history[n-3] == 1 && prev >= 2 && last == 1 { return true; }
        false
    }

    fn apply_action(action: usize, player: usize, pot: i32, mut bets: [i32; 2], mut stacks: [i32; 2], to_call: i32) -> (i32, [i32; 2], [i32; 2]) {
        if action == 1 {
            let amt = to_call.min(stacks[player]);
            bets[player] += amt; stacks[player] -= amt;
            return (pot + amt, bets, stacks);
        }
        let raise = ((pot as f32 * BET_FRACTIONS[action]) as i32).max(10);
        let total = (to_call + raise).min(stacks[player]);
        bets[player] += total; stacks[player] -= total;
        (pot + total, bets, stacks)
    }

    fn avail_actions(to_call: i32, stack: i32, history: &[u8]) -> [bool; NUM_ACTIONS] {
        let mut a = [false; NUM_ACTIONS];
        if to_call > 0 { a[0] = true; }
        a[1] = true;
        // Строгий лимит на 3 рейза за улицу, чтобы дерево не росло до миллиардов веток
        let raises = history.iter().filter(|&&act| act >= 2).count();
        if stack > to_call && raises < 3 { 
            for i in 2..NUM_ACTIONS { a[i] = true; } 
        }
        a
    }

    fn train(&mut self, et: &EvalTables, n_iter: u32, verbose: bool) {
        let mut rng = rand::thread_rng();
        let t0 = Instant::now();

        for _ in 0..n_iter {
            self.iteration += 1;
            let t = self.iteration as f32;
            let (rw, sw) = (t / (t + 1.0), (t / (t + 1.0)).powi(2));

            let mut deck = fresh_deck();
            shuffle_deck(&mut deck, &mut rng);
            let (h1, h2) = ([deck[0], deck[1]], [deck[2], deck[3]]);
            let b5 = [deck[4], deck[5], deck[6], deck[7], deck[8]];

            // МАГИЯ ЗДЕСЬ: Считаем силу руки один раз для каждой улицы, а не миллион раз!
            let mut b_p1 = [0u8; 4];
            let mut b_p2 = [0u8; 4];
            for s in 0..4u8 {
                let board = board_for_street(&b5, s);
                b_p1[s as usize] = self.hand_bucket(et, h1, board, s, &mut rng);
                b_p2[s as usize] = self.hand_bucket(et, h2, board, s, &mut rng);
            }

            for tr in 0..2u8 {
                // Передаем history как &mut Vec, чтобы избежать миллионов аллокаций памяти
                self.cfr(et, h1, h2, b5, 0, &mut Vec::new(), 15, [5, 10], [995, 990], tr, 0, rw, sw, &b_p1, &b_p2, &mut rng);
            }

            // Возвращаем на 5000, потому что теперь бот будет летать
            if verbose && self.iteration % 5000 == 0 {
                let el = t0.elapsed().as_secs_f32();
                eprintln!("  DCFR {} | {} nodes | {:.0} it/s | {:.1}s",
                    self.iteration, self.nodes.len(), self.iteration as f32 / el, el);
            }
        }
        if verbose {
            eprintln!("  Done: {} infosets, {} iters, {:.1}s",
                self.nodes.len(), self.iteration, t0.elapsed().as_secs_f32());
        }
    }

    fn cfr(&mut self, et: &EvalTables, h1: [u8; 2], h2: [u8; 2], b5: [u8; 5],
           street: u8, history: &mut Vec<u8>, pot: i32, bets: [i32; 2], stacks: [i32; 2],
           tr: u8, depth: u8, rw: f32, sw: f32, 
           b_p1: &[u8; 4], b_p2: &[u8; 4], rng: &mut impl Rng) -> f32
    {
        if depth > 16 {
            // Быстрая аппроксимация эквити, чтобы не вызывать Монте-Карло глубоко в дереве
            let hb = if tr == 0 { b_p1[street as usize] } else { b_p2[street as usize] };
            let eq = (hb as f32 + 0.5) / N_EQ_BUCKETS as f32;
            return eq * pot as f32 - bets[tr as usize] as f32;
        }

        let n = history.len();

        // Fold terminal
        if n >= 1 && history[n-1] == 0 {
            let folder = ((n-1) % 2) as u8;
            return if folder == tr { -(bets[tr as usize] as f32) }
                   else { bets[1 - tr as usize] as f32 };
        }

        if Self::is_street_done(history) {
            if street >= 3 {
                let mut m7 = [h1[0], h1[1], b5[0], b5[1], b5[2], b5[3], b5[4]];
                let r1 = et.eval7(&m7);
                m7[0] = h2[0]; m7[1] = h2[1];
                let r2 = et.eval7(&m7);
                let half = pot as f32 / 2.0;
                return if r1 < r2 { if tr == 0 { half } else { -half } }
                       else if r2 < r1 { if tr == 1 { half } else { -half } }
                       else { 0.0 };
            }
            let mut empty_hist = Vec::new();
            return self.cfr(et, h1, h2, b5, street+1, &mut empty_hist, pot, [0,0], stacks, tr, depth+1, rw, sw, b_p1, b_p2, rng);
        }

        let player = (n % 2) as u8;
        let p = player as usize;

        // Берем готовый bucket за O(1)
        let hb = if player == 0 { b_p1[street as usize] } else { b_p2[street as usize] };
        let sb = spr_bucket(pot, stacks[p]);
        let key = make_infoset_key(street, hb, sb, history);

        let to_call = (bets[1-p] - bets[p]).max(0);
        let avail = Self::avail_actions(to_call, stacks[p], history);

        let strat = {
            let node = self.nodes.entry(key).or_insert_with(CfrNode::new);
            let s = node.get_strategy();
            for a in 0..NUM_ACTIONS { node.strategy_sum[a] += sw * s[a]; }
            s
        };

        if player == tr {
            let mut utils = [0.0f32; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                if !avail[a] { continue; }
                if a == 0 { utils[a] = -(bets[p] as f32); continue; }
                let (np, nb, ns) = Self::apply_action(a, p, pot, bets, stacks, to_call);
                history.push(a as u8);
                utils[a] = self.cfr(et, h1, h2, b5, street, history, np, nb, ns, tr, depth+1, rw, sw, b_p1, b_p2, rng);
                history.pop();
            }

            let cu = utils[1]; 
            for a in 0..NUM_ACTIONS {
                if !avail[a] {
                    utils[a] = if a == 0 && to_call == 0 { cu }
                               else if a == 0 { -(bets[p] as f32) }
                               else { cu };
                }
            }

            let nu: f32 = (0..NUM_ACTIONS).map(|a| strat[a] * utils[a]).sum();
            let node = self.nodes.get_mut(&key).unwrap();
            for a in 0..NUM_ACTIONS { node.regret_sum[a] = node.regret_sum[a] * rw + (utils[a] - nu); }
            nu
        } else {
            let mut a = sample_action(&strat, rng);
            if !avail[a] { a = 1; }
            if a == 0 && to_call == 0 { a = 1; }
            if a == 0 { return bets[1 - tr as usize] as f32; }

            let (np, nb, ns) = Self::apply_action(a, p, pot, bets, stacks, to_call);
            history.push(a as u8);
            let util = self.cfr(et, h1, h2, b5, street, history, np, nb, ns, tr, depth+1, rw, sw, b_p1, b_p2, rng);
            history.pop();
            util
        }
    }

    fn get_action(&self, et: &EvalTables, hole: [u8; 2], board: &[u8], street: u8,
                  pot: i32, to_call: i32, stack: i32, history: &[u8], rng: &mut impl Rng) -> (usize, f32)
    {
        let hb = if street == 0 {
            let c = preflop_canonical(hole[0], hole[1]) as usize;
            if c < self.preflop_buckets.len() { self.preflop_buckets[c] } else { (N_EQ_BUCKETS/2) as u8 }
        } else {
            let eq = if board.len() == 5 { postflop_equity(et, hole, board, rng) }
                     else { mc_equity(et, hole, board, 300, rng) };
            equity_bucket(eq)
        };

        let key = make_infoset_key(street, hb, spr_bucket(pot, stack), history);

        let strat = match self.nodes.get(&key) {
            Some(n) => n.avg_strategy(),
            None => {
                let mut s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                s[0] = if to_call > 0 { 0.2 } else { 0.0 };
                s[1] = 0.6; s[2] = 0.1; s[3] = 0.05; s[4] = 0.03; s[5] = 0.01; s[6] = 0.01;
                let t: f32 = s.iter().sum();
                for v in &mut s { *v /= t; }
                s
            }
        };

        let mut f = strat;
        if to_call == 0 { f[0] = 0.0; }
        if stack <= to_call { for a in 2..NUM_ACTIONS { f[a] = 0.0; } }

        let total: f32 = f.iter().sum();
        if total > 0.0 { for v in &mut f { *v /= total; } }
        else { f = [0.0; NUM_ACTIONS]; f[1] = 1.0; }

        let eq = if street > 0 && !board.is_empty() { mc_equity(et, hole, board, 200, rng) } else { 0.5 };
        (sample_action(&f, rng), eq)
    }

    fn river_resolve(&mut self, et: &EvalTables, h1: [u8; 2], b5: [u8; 5], pot: i32, stacks: [i32; 2], n_iter: u32) {
        let mut rng = rand::thread_rng();
        let mut dead = [false; 52];
        dead[h1[0] as usize] = true; dead[h1[1] as usize] = true;
        for &b in &b5 { dead[b as usize] = true; }
        let live: Vec<u8> = (0..52u8).filter(|&c| !dead[c as usize]).collect();
        if live.len() < 2 { return; }

        for _ in 0..n_iter {
            self.iteration += 1;
            let t = self.iteration as f32;
            let (rw, sw) = (t / (t + 1.0), (t / (t + 1.0)).powi(2));

            let i0 = rng.gen_range(0..live.len());
            let mut i1 = rng.gen_range(0..live.len()-1);
            if i1 >= i0 { i1 += 1; }
            let h2 = [live[i0], live[i1]];

            let mut b_p1 = [0u8; 4];
            let mut b_p2 = [0u8; 4];
            for s in 0..4u8 {
                let board = board_for_street(&b5, s);
                b_p1[s as usize] = self.hand_bucket(et, h1, board, s, &mut rng);
                b_p2[s as usize] = self.hand_bucket(et, h2, board, s, &mut rng);
            }

            for tr in 0..2u8 {
                self.cfr(et, h1, h2, b5, 3, &mut Vec::new(), pot, [0,0], stacks, tr, 0, rw, sw, &b_p1, &b_p2, &mut rng);
            }
        }
    }

    fn save(&self, path: &str) -> io::Result<()> { std::fs::write(path, bincode::serialize(self).unwrap()) }
    fn load(path: &str) -> Option<Self> { bincode::deserialize(&std::fs::read(path).ok()?).ok() }
}

// ═══════════════════════════════════════════════════════════════════
//  GAME PLAY
// ═══════════════════════════════════════════════════════════════════

const SB: i32 = 5;
const BB: i32 = 10;
const BRAIN_PATH: &str = "solver_brain.bin";

fn prompt(msg: &str) -> String {
    print!("{}", msg); io::stdout().flush().unwrap();
    let mut s = String::new(); io::stdin().read_line(&mut s).unwrap();
    s.trim().to_string()
}

fn play_hand(solver: &mut Solver, et: &EvalTables, ai_stack: &mut i32, hu_stack: &mut i32) -> i8 {
    let mut rng = rand::thread_rng();
    let mut deck = fresh_deck();
    shuffle_deck(&mut deck, &mut rng);

    let (ai_hole, hu_hole) = ([deck[0], deck[1]], [deck[2], deck[3]]);
    let b5 = [deck[4], deck[5], deck[6], deck[7], deck[8]];

    let (mut ai_inv, mut hu_inv) = (SB, BB);
    let (mut pot, mut ai_st, mut hu_st) = (SB + BB, *ai_stack - SB, *hu_stack - BB);
    let mut board: Vec<u8> = Vec::new();

    let streets = [(0u8, "PREFLOP", 0usize), (1, "FLOP", 3), (2, "TURN", 4), (3, "RIVER", 5)];

    for &(si, name, board_len) in &streets {
        while board.len() < board_len { board.push(b5[board.len()]); }

        if si == 3 && ai_st > 0 && hu_st > 0 {
            solver.river_resolve(et, ai_hole, b5, pot, [ai_st, hu_st], 500);
        }

        println!("\n  ── {} ──", name);
        if !board.is_empty() { println!("  Board: {}", cards_str(&board)); }
        println!("  Pot: {}  |  Your cards: {}", pot, cards_str(&hu_hole));
        println!("  Stacks: 🤖{}  🙎{}", ai_st, hu_st);

        let order: [u8; 2] = if si == 0 { [0, 1] } else { [1, 0] };
        let mut sb = if si == 0 { [SB, BB] } else { [0, 0] };
        let mut history: Vec<u8> = Vec::new();
        let mut last_raiser: Option<u8> = if si == 0 { Some(1) } else { None };
        let mut acted = 0u8;

        for rnd in 0..6u8 {
            let who = order[(rnd % 2) as usize];
            if acted >= 2 && last_raiser.map_or(true, |lr| who == lr) { break; }

            let tc = if who == 0 { (sb[1] - sb[0]).max(0) } else { (sb[0] - sb[1]).max(0) };

            if who == 0 {
                if ai_st <= 0 { acted += 1; continue; }

                let (mut a, eq) = solver.get_action(et, ai_hole, &board, si, pot, tc, ai_st, &history, &mut rng);
                if a == 0 && tc == 0 { a = 1; }
                if a >= 2 && ai_st <= tc { a = 1; }

                match a {
                    0 => { println!("  🤖 FOLD"); *hu_stack = hu_st + pot; *ai_stack = ai_st; return -1; }
                    1 => {
                        let c = tc.min(ai_st);
                        ai_st -= c; ai_inv += c; pot += c; sb[0] += c;
                        println!("  🤖 {}", if tc == 0 { "CHECK".into() } else { format!("CALL {}", c) });
                    }
                    _ => {
                        let frac = BET_FRACTIONS[a];
                        let total = (tc + ((pot as f32 * frac) as i32).max(BB)).min(ai_st);
                        ai_st -= total; ai_inv += total; pot += total; sb[0] += total;
                        last_raiser = Some(0);
                        if total >= *ai_stack - SB { println!("  🤖 ALL-IN {}", total); }
                        else { println!("  🤖 BET {}%pot → {} (eq={:.0}%)", (frac*100.0) as i32, total, eq*100.0); }
                    }
                }
                history.push(a as u8); acted += 1;
            } else {
                if hu_st <= 0 { acted += 1; continue; }

                let mut opts = Vec::new();
                if tc > 0 { opts.push(format!("[0]Fold")); opts.push(format!("[1]Call {}", tc.min(hu_st))); }
                else { opts.push("[1]Check".into()); }
                if hu_st > tc {
                    for (i, lbl) in [(2,"33%"),(3,"50%"),(4,"75%"),(5,"pot"),(6,"150%")] {
                        opts.push(format!("[{}]Bet {}", i, lbl));
                    }
                    opts.push(format!("[7]All-in {}", hu_st));
                }
                println!("  {}", opts.join(" | "));

                loop {
                    let act: u8 = match prompt("  > ").parse() { Ok(v) => v, Err(_) => continue };
                    match act {
                        0 if tc > 0 => {
                            println!("  🙎 FOLD"); *ai_stack = ai_st + pot; *hu_stack = hu_st; return 1;
                        }
                        1 => {
                            let c = tc.min(hu_st);
                            hu_st -= c; hu_inv += c; pot += c; sb[1] += c;
                            println!("  🙎 {}", if tc == 0 { "CHECK" } else { "CALL" });
                            history.push(1); acted += 1; break;
                        }
                        2..=6 if hu_st > tc => {
                            let total = (tc + ((pot as f32 * BET_FRACTIONS[act as usize]) as i32).max(BB)).min(hu_st);
                            hu_st -= total; hu_inv += total; pot += total; sb[1] += total;
                            last_raiser = Some(1);
                            println!("  🙎 BET {}", total);
                            history.push(act); acted += 1; break;
                        }
                        7 if hu_st > 0 => {
                            let a = hu_st; hu_st = 0; hu_inv += a; pot += a; sb[1] += a;
                            last_raiser = Some(1);
                            println!("  🙎 ALL-IN {}", a);
                            history.push(6); acted += 1; break;
                        }
                        _ => println!("  ?"),
                    }
                }
            }
        }
    }

    // Showdown
    let mut m7 = [ai_hole[0], ai_hole[1], b5[0], b5[1], b5[2], b5[3], b5[4]];
    let ra = et.eval7(&m7);
    m7[0] = hu_hole[0]; m7[1] = hu_hole[1];
    let rh = et.eval7(&m7);

    println!("\n  ═══ SHOWDOWN ═══");
    println!("  🤖 {} → {}", cards_str(&ai_hole), rank_name(ra));
    println!("  🙎 {} → {}", cards_str(&hu_hole), rank_name(rh));
    println!("  Board: {}", cards_str(&b5));

    if ra < rh { println!("  🤖 WINS +{}", pot - ai_inv); *ai_stack = ai_st + pot; *hu_stack = hu_st; 1 }
    else if rh < ra { println!("  🙎 YOU WIN +{}", pot - hu_inv); *ai_stack = ai_st; *hu_stack = hu_st + pot; -1 }
    else { println!("  SPLIT"); *ai_stack = ai_st + pot/2; *hu_stack = hu_st + pot - pot/2; 0 }
}

// ═══════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════

fn main() {
    println!("╔═══════════════════════════════════════════╗");
    println!("║  POKER SOLVER — DCFR + Lookup Eval        ║");
    println!("║  7 bet sizes + River resolving            ║");
    println!("╚═══════════════════════════════════════════╝");

    eprintln!("  Building eval tables...");
    let t0 = Instant::now();
    let et = EvalTables::new();
    eprintln!("  Ready: {:.2}s | Sanity: RF rank = {} (expect 1)",
        t0.elapsed().as_secs_f32(), et.eval5(&[32, 36, 40, 44, 48]));

    let mut solver = Solver::load(BRAIN_PATH).map(|s| {
        eprintln!("  Brain loaded: {} infosets, {} iters", s.nodes.len(), s.iteration); s
    }).unwrap_or_else(|| { eprintln!("  New brain"); Solver::new(&et) });

    loop {
        println!("\n  [1] Play  [2] Train  [3] Train+Play  [q] Quit");
        let ch = prompt("  > ");
        match ch.as_str() {
            "2" | "3" => {
                let n: u32 = prompt("  DCFR iterations: ").parse().unwrap_or(50000);
                solver.train(&et, n, true);
                solver.save(BRAIN_PATH).ok();
                eprintln!("  Saved ({} nodes)", solver.nodes.len());
                if ch == "2" { continue; }
            }
            "1" => {}
            "q" | "Q" => { solver.save(BRAIN_PATH).ok(); println!("  Bye."); break; }
            _ => continue,
        }

        let (mut ai, mut hu, mut hand) = (1000i32, 1000i32, 0u32);
        loop {
            hand += 1;
            println!("\n━━━ Hand #{} | 🤖{} vs 🙎{} ━━━", hand, ai, hu);
            if prompt("  [Enter] play / [q] quit: ") == "q" { break; }
            play_hand(&mut solver, &et, &mut ai, &mut hu);
            if ai <= 0 { println!("\n  🎉 You busted the AI!"); break; }
            if hu <= 0 { println!("\n  💀 AI wins."); break; }
            if hand % 20 == 0 { solver.save(BRAIN_PATH).ok(); }
        }
        solver.save(BRAIN_PATH).ok();
        println!("  Session: 🤖{} | 🙎{} | Net: {:+}", ai, hu, hu - 1000);
    }
}