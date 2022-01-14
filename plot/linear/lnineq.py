#!/usr/bin/env python3

'''
    class to handle linear inequality constraint

    Similaryly as equality, inequality constraints are converted to
        lower and upper bound of basis vars with only previous vars
    for example
                                lc1 <= v1 <= uc1
                         l21*v1+lc2 <= v2 <= uc2+u21*v1
                  l31*v1+l32*v2+lc3 <= v3 <= uc2+u31*v1+u32*v2
               ...
    In this class, variable names are ignored,
        since it always cooperates with `LnEqs` object,
            which hold variables, and also handle possible degeneraces
    
    Additional inequality from pair of lower and upper bounds in every are
        also included in bounds of previous vars,
    in order to detect conflict and degenerate cases dynamically.

    During construction of bounds, it is maintained to be non-degenerate and non-conflict.
    All degeneraces are poped to a temporaray buffer,
        which is then handled by the cooperating `LnEqs` object
'''

from .bases import LnCoeffs, LnComb
from .tools import PrecisceComparator, NormComp, \
                   insert_iter_to_list, iter_prod, get_indent

class LnIneqs:
    '''
        base class to handle linear inequality constraints
    '''
    def __init__(self, precise=None):
        '''
            init method

            Parameters:
                precise: PrecisceComparator, float, or None
                    used to create comparator with precise
        '''
        # comparator
        if not isinstance(precise, PrecisceComparator):
            precise=PrecisceComparator(precise)
        self._comp=precise   # comparator

        # container of bounds
        self._init_bounds_container()

    def _init_bounds_container(self):
        '''
            init buffer for inequality constraints

            ineqs are stored as lower and upper bound of vars
        '''
        self._bound_vars=[]

    # copy
    def copy_ineqs_from(self, other):
        '''
            copy vars from another object

            ignore comparator
        '''
        if not isinstance(other, self.__class__):
            raise TypeError('cannot copy from other type')

        self._bound_vars.clear()
        for ls, us in other._bound_vars:
            newls=[ksc.copy() for ksc in ls]
            newus=[ksc.copy() for ksc in us]
            self._bound_vars.append((newls, newus))

    def copy(self):
        obj=self.__class__(self._comp)
        obj.copy_ineqs_from(self)

        return obj

    # containing relation
    @staticmethod
    def contains_bounds(ksc, bounds, upper=True, pop=True, comp=NormComp):
        '''
            whether new ineq contains region restricted by existed bounds

            return True if new is redundant

            At current, just do simple check
            for example
                new bound to add: v3 <= u1*v1+u2*v2+cu1
                old bound exists: v3 <= u1*v1+u2*v2+cu2
            if cu2<=cu1, then new bound is redundant

            ================

            :param pop: bool
                whether to pop old constraints,
                    if they are redundant with respect to new one

            :param comp: PreciseComparator, optional
                use `NormComp` by default
        '''
        is_redu=False

        sign=2*int(upper)-1   # -1 for lower, 1 for upper
        inds_pop=[]   # index to pop
        for i, ksci in enumerate(bounds):
            # new ineq: dk1*v1+dk2*v2+...+dc<=0
            dksc=(ksci-ksc)*sign
            dks, dc=dksc

            inds_nonz=comp.inds_nonzero(dks)
            if len(inds_nonz)!=0:
                continue

            if comp.is_le_zero(dc): # redundant ineq, skip
                is_redu=True
                continue

            inds_pop.append(i)

        if pop:
            for i in reversed(inds_pop):  # pop redundant ineqs
                bounds.pop(i)

        return is_redu

    # add new vars
    def insert_new_vars(self, ind, num):
        '''
            insert new vars before index `ind`
        '''
        if num<=0:
            return

        # extend ksc of bounds for subsequent vars
        zeros=[0]*num
        for lowers, uppers in self._bound_vars[ind:]:
            for bounds in [lowers, uppers]:
                n=len(bounds)
                for i in range(n):
                    bounds[i].insert_zeros(ind, num)

        # new bounds for new vars
        newbds=[([], []) for _ in range(num)]
        insert_iter_to_list(self._bound_vars, ind, newbds, inplace=True)

    def append_new_vars(self, num):
        '''
            append new vars in the end
        '''
        if num<=0:
            return

        self._bound_vars.extend([([], []) for _ in range(num)])

    # add new bound
    def add_new_bound(self, ksc, upper=True):
        '''
            add new bound:
                x_(i+1) <= k1*v1+...+ki*vi+c or
                x_(i+1) >= k1*v1+...+ki*vi+c

            Parameters:
                ksc: LnCoeffs
                    coeffs and bias of the combination

                    `len(coeffs)` is the index of var to bound

                upper: bool
                    whether upper or lower bound

            Return:
                dict of LnCoeffs, using len as the key
                    each for a degenerated combination
        '''
        ind=len(ksc)
        assert 0<=ind<len(self._bound_vars), 'index out of range'

        ilu=int(upper)  # 0 for lower, 1 for upper
        sign=2*ilu-1    # -1 for lower, 1 for upper
        bounds=self._bound_vars[ind][ilu]    # dest buffer of bounds

        # buffer to store degeneraces
        buf_degens=self._new_degen_buffer()

        # containing check in dest bounds
        if self._pop_if_contains(ksc, bounds, upper):
            return buf_degens

        # sub-ineqs from bounds of opposite side: li(x1, ..) <= uj(x1, ..)
        bdsopp=self._bound_vars[ind][1-ilu]  # opposite bounds

        if not bdsopp:
            bounds.append(ksc.copy())
            return buf_degens

        ## check degenerate and construct sub-ineq
        dkscs=[]
        for ksci in bdsopp:
            # to standard form: dk1*v1+dk2*v2+...+dc<=0
            dksc=(ksci-ksc)*sign
            dks, dc=dksc

            inds_nonz=self._comp.inds_nonzero(dks)
            if len(inds_nonz)==0:
                if not self._comp.is_le_zero(dc):  # conflict
                    raise LnConflictError('conflict in inequality')

                if self._comp.is_zero(dc):  # degenerate
                    if ind not in buf_degens:
                        buf_degens.add_degen(ksc)
                continue

            ilz=inds_nonz[-1]
            dkscs.append(dksc[:(ilz+1)])

        ## add sub-ineq
        for ksci in dkscs:
            ki=ksci.pop()
            ksci=-ksci/ki

            ui=self._comp.is_ge_zero(ki)
            buf_degens.join_degens(self.add_new_bound(ksci, upper=ui))

        ## if degenerate, clear bounds, since ineqs from li<=ui already added
        if buf_degens:
            bdsopp.clear()

            if ind not in buf_degens:  # degenerate if sub-ineq degenerate
                buf_degens.add_degen(ksc)
            else:
                # degenerate in current var, clear all bounds,
                #     since sub-ineqs added before
                bounds.clear()
        else:
            bounds.append(ksc.copy())

        return buf_degens

    ## auxiliary functions
    def _new_degen_buffer(self):
        '''
            return new degen buffer
                with current comparator
        '''
        return DegenBuffer(self._comp)

    def _pop_if_contains(self, ksc, bounds, upper):
        '''
            wrapper of `contains_bounds`
                with pop=True and comp=self._comp
        '''
        return self.contains_bounds(ksc, bounds, upper=upper,
                                pop=True, comp=self._comp)

    # adjust vars order
    def mv_var_to_end(self, ind):
        '''
            move var at index `ind` to end
                keep 3 conditions of ineqs after moving
                    - bounds expressed by previous vars
                    - additional ineqs from li<=uj included
                    - non-degenerate

            ignore order of vars in `LnEqs`
                to resort them in other method

            :param ind: int
                index of var to move
        '''
        n=len(self._bound_vars)
        ind=range(n)[ind]

        if ind==n-1:  # already last var
            return

        # move old bds to end
        bdsold=self._bound_vars.pop(ind)
        for ksc in [*bdsold[0], *bdsold[1]]:
            ksc.fill_to_len(n-1)
        self._bound_vars.append(bdsold)

        # new bounds from subsequent vars
        bdsnew=([], [])
        for lus in self._bound_vars[ind:-1]:
            for i, bdsi in enumerate(lus):
                s=2*i-1   # -1 for lower, 1 for upper
                for j in reversed(range(len(bdsi))):
                    kscj=bdsi[j]

                    k0=kscj.pop(ind)
                    if self._comp.is_zero(k0): # no new ineq for var `ind`
                        continue

                    # new ineq
                    ## to standard form: k0*v <= k1*v1+...-vi+c
                    kscj.append(-1)
                    kscj*=s
                    k0*=-s

                    ik0=int(self._comp.is_ge_zero(k0))
                    bdsnew[ik0].append(kscj/k0)

                    # pop from bounds
                    bdsi.pop(j)

        # until this step, 3 conditions still satisfied
        #   except, some additional ineqs need to be added

        # add new bounds
        for i, bds in enumerate(bdsnew):
            for kscj in bds:
                degens=self.add_new_bound(kscj, upper=i)

                # should not have degenerace from mathematics
                #     if non-degenerate before moving
                if degens:
                    raise Exception('unexpected degeneraces in mathematics')

    # pop degenerate vars
    def pop_var_for_degen(self, ksc):
        '''
            remove var for degenerate
            return additional degeneraces raised by it

            :param ksc: LnCoeffs
                coeffs and bias of the degenerate var
        '''
        # buffer for new degeneraces
        buf_degens=self._new_degen_buffer()
        buf_degens.add_degen(ksc)

        ind=len(ksc)  # index of degenerate var

        # add `ksc` as both lower and upper bound
        if self._bound_vars[ind][0]:
            newds=self.add_new_bound(ksc, upper=True)
            buf_degens.join_degens(newds)

        if self._bound_vars[ind][1]:
            newds=self.add_new_bound(ksc, upper=False)
            buf_degens.join_degens(newds)

        buf_degens.pop_degen(ind)

        # pop degenerate bounds
        self._bound_vars.pop(ind)
        # assert not (ls+us), 'unexpected non-empty bounds'

        # handle subsequent vars
        for lus in self._bound_vars[ind:]:
            for i, bdsi in enumerate(lus):
                # remove ineqs contained by others
                for j in reversed(range(len(bdsi))):
                    kscj=bdsi[j]

                    k0=kscj.pop(ind)
                    kscj[:ind]+=ksc*k0

                    if self._pop_if_contains(kscj, bdsi[(j+1):], False):
                        bdsi.pop(j)
                        continue

            # no need to add sub-ineqs from li<=uj
            #     which have been involved by condition 2

            # handle degenerate
            lowers, uppers=lus
            for ksc0, ksc1 in iter_prod(uppers, lowers):
                dksc=ksc1-ksc0
                dks, dc=dksc

                inds_nonz=self._comp.inds_nonzero(dks)
                if len(inds_nonz)==0:
                    # should not have conflict, already involved before
                    assert self._comp.is_le_zero(dc)

                    if self._comp.is_zero(dc):  # degenerate
                        buf_degens.add_degen(ksc0)

                        lowers.clear()
                        uppers.clear()

                        break
        return buf_degens

    def pop_vars_for_degens(self, degens):
        '''
            pop degenerate vars
            return list of ksc for each treatment in order
                which means linear eqs of previous vars

            Parameters:
                degens: DegenBuffer
                    with buffer {len(ksc): ksc}
        '''
        eqs=[] # linear eqs of previous vars
        while degens:
            ksc=degens.pop_degen()

            # handle one degenerace
            newds=self.pop_var_for_degen(ksc)
            degens.join_degens(newds)
            
            eqs.append(ksc)
        return eqs

    # cooperate with LnEqs for degeneraces
    def _coop_with_eqs_for_degens(self, eqs, degens):
        '''
            cooperate with LnEqs for degeneraces,
                absorbing degens to `eq`, and
                poping vars in `self`

            :param eqs: LnEqs
                object holding basis variables

            :param degens: DegensBuffer
        '''
        # assert same state for both ineqs and eqs
        if eqs.numbasis!=len(self._bound_vars):
            raise ValueError('mismatch between eqs and ineqs')

        for ksci in self.pop_vars_for_degens(degens):
            ind=len(ksci)

            n=eqs.numbasis-1
            ksci.fill_to_len(n)

            eqs.add_eq_to_basis_var(ind, *ksci)

    ## add new bound
    def add_new_bound_coop_with(self, eqs, ksc, upper=True):
        '''
            cooperate with LnEqs to add new bound
        '''
        degens=self.add_new_bound(ksc, upper)

        # cooperate with LnEqs
        self._coop_with_eqs_for_degens(eqs, degens)

    ## add eq constraint
    def add_degen_coop_with(self, eqs, ksc):
        '''
            cooperate with LnEqs to add degenerate bounds
        '''
        degens=self._new_degen_buffer()
        degens.add_degen(ksc)

        # handle new degens
        self._coop_with_eqs_for_degens(eqs, degens)

    # evaluate bounds
    def eval_bounds_of(self, ksc):
        '''
            eval lower and upper bounds
                for a linear combination

            :param ksc: LnCoeffs-like
        '''
        ks, c=LnCoeffs.lncoeffs(ksc)
        if self._comp.all_zeros(ks):
            return c, c

        ilz=self._comp.inds_nonzero(ks)[-1]
        ks=ks[:(ilz+1)]

        # add bounds one var by one var
        lubounds=([LnCoeffs(ks, c)], [LnCoeffs(ks, c)])  # init bounds

        for lubsi in reversed(self._bound_vars[:(ilz+1)]):
            oldlubds=[(0, ksc) for ksc in lubounds[0]]+\
                     [(1, ksc) for ksc in lubounds[1]] 
            newlubds=([], [])
            while oldlubds:
                ind, ksc=oldlubds.pop()
                k0=ksc.pop()

                if self._comp.is_zero(k0):
                    self._add_if_not_contains(ksc, newlubds[ind], ind)
                    continue

                # eliminate last vars
                # from its bounds
                ik=int(self._comp.is_ge_zero(k0))
                s0=2*ik-1
                bsi=lubsi[(1-ik)+s0*ind]
                # bsi=lubsi[ind] if self._is_ge_zero(k0) else lubsi[1-ind]

                for ksci in bsi:
                    self._add_if_not_contains(ksc+k0*ksci, newlubds[ind], ind)

                ## from old bounds
                s=2*ind-1       # -1 for lower, 1 for upper
                ksc=s*ksc       # to s*vj <= k1*v1 +...+c

                k0, s0=s*k0, s*s0
                for j, kscj in oldlubds:
                    sj=2*j-1
                    kscj=sj*kscj  # to s*vj <= k1*v1 +...+c
                    kj=kscj.pop()

                    if self._comp.is_ge_zero(kj*k0):
                        continue

                    k1=-s*kj+sj*k0
                    if self._comp.is_zero(k1):
                        continue

                    ksc1=(-ksc*kj+kscj*k0)/k1
                    ik1=int(self._comp.is_ge_zero(k1*s0))
                    self._add_if_not_contains(ksc1, newlubds[ik1], ik1)

            lubounds=newlubds

        # lower and upper
        lu=[]
        for bds in lubounds:
            if not bds:
                lu.append(None)
                continue
            lu.append(bds[0].const)

        return tuple(lu)

    ## auxiliary functions
    def _add_if_not_contains(self, ksc, buf, upper):
        '''
            add new bound to a buffer
                when it does not contain old restrictions
        '''
        if not self._pop_if_contains(ksc, buf, upper):
            buf.append(ksc)

    # to string
    def get_lines_of_ineqs(self, vs=None):
        '''
            return lines of ineqs
                each for a bound

            :param vs: list of variables
                if None, use a, b, c,...
                             or v0, v1, ...
        '''
        if not self._bound_vars:
            return []

        n=len(self._bound_vars)
        if vs is None:
            vs=default_vs(n)
        elif len(vs)!=n:
            raise ValueError('mismatch for len of `vs`')

        lines=[]

        enumer=reversed(list(enumerate(self._bound_vars)))
        for i, (lowers, uppers) in enumer:
            vi=vs[i]
            vsi=vs[:i]
            for bounds, s in [(lowers, '>='), (uppers, '<=')]:
                for (ks, c) in bounds:
                    *ks, c=self._comp.filter([*ks, c])

                    lines.append('%s %s %s' % (vi, s, LnComb(vsi, ks, c)))

        return lines

    def get_str_of_ineqs(self, vs=None, indent=None):
        '''
            str of eqs
        '''
        lines=self.get_lines_of_ineqs(vs)
        if indent:
            indent=get_indent(indent)
            lines=[indent+l for l in lines]
        return '\n'.join(lines)

    def __str__(self):
        return self.get_str_of_ineqs()

    def info(self, indent=None, vs=None):
        print(self.get_str_of_ineqs(vs, indent))


# auxiliary functions
class DegenBuffer:
    '''
        buffer to store degeneraces
    '''
    def __init__(self, comp=NormComp):
        '''
            init

            :param comp: PrecisceComparator
        '''
        self._comp=comp

        self._degens={}

    # add degenerace
    def add_degen(self, ksc):
        '''
            add a degenerate combination to buffer
                which uses len(ks) as key

            :param ksc: LnCoeffs
                ks, c of degenerate combination

                If ks=[k1, k2, ..., ki], then equality to add is
                    v_(i+1)=k1*v1+...+ki*vi+c
        '''
        buf=self._degens
        n=len(ksc)  # actually len(ksc.coeffs)

        if n not in buf:
            buf[n]=ksc.copy()
            return

        dksc=ksc-buf[n]
        dks, dc=dksc

        inds_nonz=self._comp.inds_nonzero(dks)
        if len(inds_nonz)==0:
            if not self._comp.is_zero(dc):  # conflict
                raise LnConflictError('conflict from degenerace')

            return

        ilz=inds_nonz[-1]
        klz=dksc[ilz]
        self.add_degen(dksc[:ilz]/klz)

    def join_degens(self, degens):
        '''
            join multi-degeneraces

            :param degens: dict-like or DegenBuffer
        '''
        for n in degens:
            self.add_degen(degens[n])

    # pop degen
    def pop_degen(self, n=None):
        '''
            pop a degenerate from buffer

            `n` is len of coeffs
                if None, pop ksc with max len
        '''
        buf=self._degens

        if n is None:
            n=max(buf.keys())
        return buf.pop(n)

    # work like dict
    def __getitem__(self, v):
        return self._degens[v]

    def __iter__(self):
        return iter(self._degens.keys())

    def __contains__(self, k):
        return k in self._degens

    def __bool__(self):
        return bool(self._degens)

    # to str
    def get_lines_of_degens(self, vs=None):
        '''
            lines for degeneraces
        '''
        ns=sorted(self._degens.keys())
        if not ns:
            return []

        n=max(ns)+1
        if vs is None:
            vs=default_vs(n)
        elif len(vs)!=n:
            raise ValueError('mismatch for len of `vs`')

        lines=[]
        for i in reversed(ns):
            ks, c=self._degens[i]
            *ks, c=self._comp.filter([*ks, c])

            s='%s = %s' % (vs[i], LnComb(vs[:i], ks, c))
            lines.append(s)
        return lines

    def get_str_of_degens(self, indent=None, vs=None):
        '''
            str of eqs
        '''
        lines=self.get_lines_of_degens(vs)
        if indent:
            indent=get_indent(indent)
            lines=[indent+l for l in lines]
        return '\n'.join(lines)

    def __str__(self):
        return self.get_str_of_degens()

    def info(self, indent=None, vs=None):
        print(self.get_str_of_degens(vs=vs, indent=indent))

# auxiliary functions
def default_vs(n):
    '''
        default vars
    '''
    if n<=26:
        return [chr(i+ord('a')) for i in range(n)]
    return ['v%i' % i for i in range(n)]