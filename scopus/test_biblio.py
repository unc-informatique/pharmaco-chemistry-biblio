# pylint: skip-file
# %%
import pytest
import biblio_extractor as dl

COMPOUNDS = [
    ("alkaloid", "acridine"),
    ("phenolic compound", "anthraquinone"),
    ("terpenoid/terpene", "tetraterpene/carotenoid"),
    ("terpenoid/terpene", "triterpene"),
]

ACTIVITIES = [
    ("allelopathy", "germination"),
    ("allelopathy", "herbicidal"),
    ("pharmaco", "cytotoxicity"),
    ("pharmaco", "sedative"),
]


REF_COMPOUNDS_2_OR = '((KEY("acridine")) OR (KEY("anthraquinone")))'
REF_ACTIVITIES_2_OR = '((KEY("germination")) OR (KEY("herbicidal")))'
REF_COMPOUNDS_2_AND = '((KEY("acridine")) AND (KEY("anthraquinone")))'
REF_COMPOUND_1 = '((KEY("acridine")))'


# NOTE : le last parameter of dl.Query (kind) is not used by dl.build_clause (its juste forwarded)

class TestClausalQuery:
    def test_compound_one(self):
        res = dl.build_clause(dl.Query(COMPOUNDS[:1], [], [], [], None))
        assert res == REF_COMPOUND_1

    def test_compound_split(self):
        res = dl.build_clause(dl.Query(COMPOUNDS[2:3], [], [], [], None))
        assert res == '((KEY("tetraterpene") OR KEY("carotenoid")))'

    def test_compound_two(self):
        res = dl.build_clause(dl.Query(COMPOUNDS[:2], [], [], [], None))
        assert res == REF_COMPOUNDS_2_OR

    def test_activities(self):
        res = dl.build_clause(dl.Query([], ACTIVITIES[:2], [], [], None))
        assert res == REF_ACTIVITIES_2_OR

    def test_compounds_and_activities(self):
        res = dl.build_clause(dl.Query(COMPOUNDS[:2], ACTIVITIES[:2], [], [], None))
        assert res == f"{REF_COMPOUNDS_2_OR} AND {REF_ACTIVITIES_2_OR}"

    def test_pos_kw_empty(self):
        res = dl.build_clause(dl.Query([], [], COMPOUNDS[:1], [], None))
        assert res == REF_COMPOUND_1

    def test_pos_kw_empty_2(self):
        res = dl.build_clause(dl.Query([], [], COMPOUNDS[:2], [], None))
        assert res == REF_COMPOUNDS_2_AND

    def test_pos_kw(self):
        res = dl.build_clause(dl.Query(COMPOUNDS[:2], ACTIVITIES[:2], COMPOUNDS[0:1], [], None))
        assert res == f"{REF_COMPOUNDS_2_OR} AND {REF_ACTIVITIES_2_OR} AND {REF_COMPOUND_1}"

    def test_neg_kw_only(self):
        with pytest.raises(IndexError):
            dl.build_clause(dl.Query([], [], [], COMPOUNDS[:1], None))

    def test_neg_kw_nopos(self):
        res = dl.build_clause(dl.Query(COMPOUNDS[:2], ACTIVITIES[:2], [], COMPOUNDS[0:1], None))
        assert res == f"{REF_COMPOUNDS_2_OR} AND {REF_ACTIVITIES_2_OR} AND NOT {REF_COMPOUND_1}"

    def test_neg_kw_nopos_2(self):
        res = dl.build_clause(dl.Query([], [], COMPOUNDS[:2], ACTIVITIES[:2], None))
        assert res == f"{REF_COMPOUNDS_2_AND} AND NOT {REF_ACTIVITIES_2_OR}"
