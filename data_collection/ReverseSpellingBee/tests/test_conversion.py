import pytest
from convert_ipa import ipa_to_oed_with_stress, ipa_oed_mapping

# Test cases: each tuple contains the IPA transcription and the expected OED respelling.
test_cases = [
    ("/pɚˈsɛnt/", "pur-SENT"),
    ("/ˌwʌnˌoʊˈwʌn/", "wun·-oh·-WUN"),
    ("/ˌtɛn ˈdaʊnɪŋ stɹiːt/", "ten·-DOWNING-street"),
    ("/ˌtɛˈnɛks dɪˈvɛ.lə.pɚ/", "te·-NEKS-di-VE-luh-pur"),
    ("/sɪksˈtiːn-sɛl/", "siks-TEEN-sel"),
    ("/fɚst/", "furst"),
    ("/θri.sɪks.tiˈnoʊ.skoʊp/", "three-siks-tee-NOH-skohp"),
    ("/təˈmɑː.təʊ/", "tuh-MAH-toh"),
    ("/təˈmeɪ.toʊ/", "tuh-MAY-toh"),
    ("/əˈbɹi.vi.əˌt͡ʃʊɹ/", "uh-BREE-vee-uh-chuur·"),
    ("/ˌæb.əˈɹeɪ.ʃn̩/", "ab·-uh-RAY-shn"),
    ("/æbˈd͡ʒʊɹ.əˌtɔɹ.i/", "ab-JUUR-uh-tawr·-ee"),
    ("[əˈkʰɔɹɾɨ̞nʔɫi]", "uh-KHAWRRINLEE"),
    ("/ˌæk.əˈdɛm.əˌsɪz.m̩/", "ak·-uh-DEM-uh-siz·-m"),
    ("/əˈklaɪ.məˌtaɪz/", "uh-KLIGH-muh-tighz·"),
    ("[əˌkɹʌɾ.ɪɾˈeː.ʃən]", "uh-krur·-ir-AY-shuhn"),
    ("/zu(ː)m lɛnz/", "zoom-lenz"),
    ("[əˌkɹʌɾ.ɪɾˈeː.ʃən]", "uh-krur·-ir-AY-shuhn"),
    ("/fɝst/", "furst"),
    ("/əˈmɪɡ.də.lə/", "uh-MIG-duh-luh"),
    ("/əbˈstɜː(ɹ).sɪv/", "uhb-STUR-siv"),
    ("/əˈsɜ(ɹ)bɪk/", "uh-SURBIK"),
    ("/əˈsɜːbɪti/", "uh-SURBITEE"),
    ("/əˈmɜːs/", "uh-MURS"),
]

@pytest.mark.parametrize("ipa, expected", test_cases)
def test_ipa_to_oed_conversion(ipa, expected):
    result = ipa_to_oed_with_stress(ipa, ipa_oed_mapping)
    assert result == expected, f"For IPA {ipa!r}, expected {expected!r} but got {result!r}"

if __name__ == "__main__":
    pytest.main()