import numpy as np
from scipy import stats

from aciq.statistics import Gaussian

SEED: int = 42
TEST_MU_SIGMA: list[tuple[float, float]] = [(-3.0, 0.1), (0.0, 1.0), (100.0, 50.0)]
DISTRIBUTION_SAMPLE_SIZE: int = 50_000


def make_data(mu:float=2.5, sigma:float=1.3, n:int=10_000, seed:int=SEED):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


class TestGaussian:
    def test_fit_matches_scipy(self):
        for mu, sigma in TEST_MU_SIGMA:
            data = make_data(mu=mu, sigma=sigma, n=DISTRIBUTION_SAMPLE_SIZE)
            g = Gaussian(data)
            scipy_mu, scipy_sigma = stats.norm.fit(data)
            np.testing.assert_equal(g.mu, scipy_mu)
            np.testing.assert_equal(g.sigma, scipy_sigma)

    def test_pdf_matches_scipy(self):
        for mu, sigma in TEST_MU_SIGMA:
            data = make_data(mu=mu, sigma=sigma, n=DISTRIBUTION_SAMPLE_SIZE)
            g = Gaussian(data)
            scipy_mu, scipy_sigma = stats.norm.fit(data)
            np.testing.assert_equal(g.pdf(), stats.norm.pdf(data, loc=scipy_mu, scale=scipy_sigma))

    def test_logpdf_matches_scipy(self):
        for mu, sigma in TEST_MU_SIGMA:
            data = make_data(mu=mu, sigma=sigma, n=DISTRIBUTION_SAMPLE_SIZE)
            g = Gaussian(data)
            scipy_mu, scipy_sigma = stats.norm.fit(data)
            expected = stats.norm.logpdf(data, loc=scipy_mu, scale=scipy_sigma)
            np.testing.assert_equal(g.logpdf(), expected)
