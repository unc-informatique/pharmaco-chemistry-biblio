# sanity check #1
assert dataset.shape == (2 * len(all_compounds), 2 * len(all_activities))
dataset


# sanity check : group by summing on level 2 on both rows and cols produce a matrix of constants : the number of papers
submatrix_sum = dataset.groupby(level=1).sum().groupby(level=1, axis=1).sum()
number_of_papers = np.unique(submatrix_sum.values)
# if the Scopus collection did not evolve during while querying
assert len(number_of_papers) == 1

number_of_papers = number_of_papers[0]
print(f"The domain contains {number_of_papers} papers")

with_with_total = with_with_matrix.values.sum()
print(
    f"Total number of positive/positive occurrences is {with_with_total} for {number_of_papers} papers (average={with_with_total/number_of_papers})"
)







margin_cols_manual = dataset.groupby(level=1).sum().drop_duplicates().reset_index(drop=True)
margin_cols_manual.index = pd.MultiIndex.from_tuples([bex.MARGIN_IDX])
assert np.all(margin_cols.values == margin_cols_manual.values)

margin_rows_manual = dataset.groupby(level=1, axis=1).sum().iloc[:, 0]
margin_rows_manual.name = bex.MARGIN_IDX
assert np.all(margin_rows.values == margin_rows_manual.values)






# redimension the values to a 4D array
C, A = len(all_compounds), len(all_activities)
print(C, A)
M_2_2 = np.moveaxis(dataset.values.reshape((C, 2, A, 2)), 1, -2)
# 1D is easier for apply_along_axis
M_4 = M_2_2.reshape((C * A, 4))

print(f"{M_2_2.shape = }")
print(f"{M_4.shape = }")

# M.sum(axis=(2,3)) or similarly
np_nb_paper = np.sum(M_2_2, axis=(2, 3), keepdims=False)
print(f"{np.all(np_nb_paper == number_of_papers) = } (with {number_of_papers = })")




THRESHOLD = number_of_papers//1000
print(f"{THRESHOLD = }")
mask_ff = (M_4 >= THRESHOLD).all(axis=1)
mask_4 = np.repeat(mask_ff, 4, axis = 0).reshape((C,A,2,2))
mask_CA = np.moveaxis(mask_4,2,1).reshape((C*2, A*2))








# sanity checks
def sanity_scores(all_dfs):
    for metric_name, df in all_dfs.items():
        # The input matrix: the dataset after a metric is applied
        M = df.values
        N = np.sum(M)
        # Z is M normalised
        Z = M / N
        print(f"{metric_name} (grand total {N = })")

        R = np.sum(M, axis=1)  # X @ np.ones(X.shape[1])
        C = np.sum(M, axis=0)  # Z @ np.ones(X.shape[0])

        print(f"    {np.sum(R) = :.4f} and {np.sum(C) = :.4f}")

        r = np.sum(Z, axis=1)
        c = np.sum(Z, axis=0)
        # print(f"    {r[:, np.newaxis].shape = } {c[:, np.newaxis].shape = }")
        print(
            f"    {np.sum(r) = :.4f} and {np.sum(c) = :.4f}"
        )  # \n{100*r = } \n{100*c = }


sanity_scores(score_df)