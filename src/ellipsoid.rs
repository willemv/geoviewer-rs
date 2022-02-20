#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Ellipsoid {
    pub semimajor_axis: f64,
    pub semiminor_axis: f64,
    pub inverse_flattening: f64,
    pub eccentricity: f64,
}

pub fn new(semimajor_axis: f64, inverse_flattening: f64) -> Ellipsoid {
    let f = 1.0 / inverse_flattening;
    Ellipsoid {
        semimajor_axis,
        semiminor_axis: (semimajor_axis * ((inverse_flattening - 1.0) / inverse_flattening)),
        inverse_flattening,
        eccentricity: (2.0 * f - f.powi(2)).sqrt(),
    }
}

pub const fn wgs84() -> Ellipsoid {
    // we can't use float calculations in const fn's, so just fill in the precomputed values here
    // there is a test down below to make sure these don't get out of sync
    Ellipsoid {
        semimajor_axis: 6_378_137.0,
        semiminor_axis: 6_356_752.314_245_179,
        inverse_flattening: 298.257_223_563,
        eccentricity: 0.081_819_190_842_621_49,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create_wgs84() {
        let const_ellipsoid = wgs84();
        let compd_ellipsoid = new(
            const_ellipsoid.semimajor_axis,
            const_ellipsoid.inverse_flattening,
        );

        assert!(
            (const_ellipsoid.semiminor_axis - compd_ellipsoid.semiminor_axis).abs() < f64::EPSILON
        );
        assert!((const_ellipsoid.eccentricity - compd_ellipsoid.eccentricity).abs() < f64::EPSILON);
    }
}
