import yfinance as yf
import pandas as pd
import sqlite3
import time
from datetime import datetime

# --- KONFIGURASI DATABASE ---
DB_NAME = "market_data.db"

# --- DAFTAR SAHAM LENGKAP (RAW) ---
# Saya simpan list murni kode saham, nanti script akan otomatis tambah .JK
RAW_TICKERS = [
    'AALI', 'ABBA', 'ABDA', 'ABMM', 'ACES', 'ACST', 'ADCP', 'ADES', 'ADHI', 'ADMF', 'ADMG', 'ADRO', 
    'AGAR', 'AGII', 'AIMS', 'AKKO', 'AKPI', 'AKRA', 'AKSI', 'ALDO', 'ALKA', 'ALMI', 'ALTO', 'AMAR', 
    'AMFG', 'AMIN', 'AMMN', 'AMOR', 'AMRT', 'ANDI', 'ANJT', 'ANTM', 'APEX', 'APIC', 'APII', 'APLI', 
    'APLN', 'ARFI', 'ARGO', 'ARII', 'ARKA', 'ARNA', 'ARTA', 'ARTI', 'ARTO', 'ASBI', 'ASDM', 'ASGR', 
    'ASII', 'ASJT', 'ASLC', 'ASMI', 'ASPI', 'ASRI', 'ASRM', 'ASSA', 'ATIC', 'AUTO', 'AVIA', 'AYLS',
    'BABP', 'BACA', 'BAJA', 'BALI', 'BANK', 'BAPA', 'BAPI', 'BAUT', 'BAYU', 'BBCA', 'BBHI', 'BBKP', 
    'BBLD', 'BBMD', 'BBNI', 'BBRI', 'BBRM', 'BBSI', 'BBSS', 'BBTN', 'BBYB', 'BCAP', 'BCIC', 'BCIP', 
    'BDMN', 'BEBS', 'BEEF', 'BELL', 'BESS', 'BEST', 'BFIN', 'BGTG', 'BHIT', 'BIKA', 'BIMA', 'BINA', 
    'BIPI', 'BIPP', 'BIRD', 'BISI', 'BJBR', 'BJTM', 'BKDP', 'BKSL', 'BKSW', 'BLTA', 'BLTZ', 'BLUE', 
    'BMAS', 'BMHS', 'BMRI', 'BMSR', 'BMTR', 'BNBA', 'BNBR', 'BREN', 'BNGA', 'BNII', 'BNLI', 'BOGA', 
    'BOLA', 'BOLT', 'BPII', 'BRAM', 'BRIS', 'BRMS', 'BRNA', 'BRPT', 'BSDE', 'BSIM', 'BSSR', 'BSWD', 
    'BTEK', 'BTEL', 'BTON', 'BTPN', 'BTPS', 'BUKK', 'BULL', 'BUMI', 'BUVA', 'BVIC', 'BWPT', 'BYAN',
    'CAKK', 'CAMP', 'CANI', 'CARE', 'CASS', 'CASH', 'CBMF', 'CCSI', 'CEKA', 'CENT', 'CFIN', 'CINT', 
    'CITA', 'CITY', 'CLAY', 'CLEO', 'CLPI', 'CMNP', 'CMPP', 'CMRY', 'CNKO', 'CNTX', 'COCO', 'COWL', 
    'CPIN', 'CPRO', 'CSAP', 'CSIS', 'CSMI', 'CSPD', 'CSRA', 'CTBN', 'CTRA', 'CTTH', 'CUAN',
    'DART', 'DAYA', 'DEAL', 'DEFI', 'DEUD', 'DEWA', 'DGIK', 'DIDA', 'DIGI', 'DILD', 'DIVA', 'DKFT', 
    'DLTA', 'DMMX', 'DMND', 'DNAR', 'DOID', 'DPNS', 'DSFI', 'DSNG', 'DSSA', 'DUTI', 'DYAN',
    'EAST', 'ECII', 'EDGE', 'EKAD', 'ELSA', 'ELTY', 'EMDE', 'EMTK', 'ENRG', 'EPMT', 'ERAA', 'ERTX', 
    'ESSA', 'ESTA', 'ESTI', 'ETWA', 'EXCL', 
    'FAST', 'FASW', 'FILM', 'FIMP', 'FIRE', 'FISH', 'FITT', 'FMII', 'FOOD', 'FPNI', 'FREN', 
    'GAMA', 'GDST', 'GDYR', 'GEMA', 'GEMS', 'GGRM', 'GHPY', 'GIAA', 'GJTL', 'GLOB', 'GLVA', 'GMTD', 
    'GOLD', 'GOLL', 'GOTO', 'GPRA', 'GSMF', 'GTBO', 'GWSA', 'GZCO',
    'HADE', 'HAIS', 'HDFA', 'HDIT', 'HDTX', 'HEAL', 'HELI', 'HERO', 'HEXA', 'HITS', 'HMSP', 'HOKI', 
    'HOME', 'HOPE', 'HOTL', 'HRME', 'HRTA', 'HRUM', 
    'IATA', 'IBST', 'ICBP', 'ICON', 'IDEA', 'IDPR', 'IFII', 'IFSH', 'IGAR', 'IIKP', 'IKAI', 'IKBI', 
    'IMAS', 'IMJS', 'IMPC', 'INAF', 'INAI', 'INCF', 'INCI', 'INCO', 'INDF', 'INDR', 'INDS', 'INDX', 
    'INDY', 'INKP', 'INOV', 'INPC', 'INPP', 'INPS', 'INRA', 'INRU', 'INST', 'INTA', 'INTP', 'IPCC', 
    'IPCM', 'IPOL', 'IPPE', 'IPTV', 'IRRA', 'ISAT', 'ISPC', 'ISSP', 'ITIC', 'ITMA', 'ITMG',
    'JAST', 'JAWA', 'JAYA', 'JECC', 'JGLE', 'JIHD', 'JKON', 'JMAS', 'JOTRC', 'JPFA', 'JRPT', 'JSKY', 
    'JSMR', 'JSPT', 'JTPE', 
    'KAEF', 'KARW', 'KAZA', 'KBLI', 'KBLM', 'KBLV', 'KBRI', 'KDSI', 'KEEN', 'KEJU', 'KIAS', 'KICI', 
    'KIJA', 'KINS', 'KIOS', 'KKGI', 'KLBF', 'KLVF', 'KMDS', 'KMER', 'KOIN', 'KOKA', 'KONI', 'KOPI', 
    'KOTA', 'KPAL', 'KPIG', 'KRAH', 'KRAS', 'KREN', 'KRYA', 'KUAS', 'KULA',
    'LAND', 'LAPD', 'LCKM', 'LCKP', 'LDAZ', 'LEAD', 'LINK', 'LION', 'LMSH', 'LPCK', 'LPFF', 'LPGI', 
    'LPIN', 'LPKR', 'LPLI', 'LPPF', 'LPPS', 'LRNA', 'LSIP', 'LTLS', 'LUCK', 'LUCY',
    'MABA', 'MAGP', 'MAIN', 'MAMI', 'MAPA', 'MAPI', 'MARI', 'MARK', 'MASB', 'MAYA', 'MBAP', 'MBMA', 
    'MBSS', 'MBTO', 'MCAS', 'MCHA', 'MCOR', 'MDIA', 'MDKA', 'MDKI', 'MDLN', 'MDRN', 'MEDC', 'MEGA', 
    'MERK', 'META', 'MFIN', 'MFMI', 'MGLV', 'MGNA', 'MGRO', 'MICE', 'MIDI', 'MIKA', 'MINA', 'MIRA', 
    'MITI', 'MKNT', 'MKPI', 'MLBI', 'MLIA', 'MLPL', 'MLPT', 'MMLP', 'MNCN', 'MOLI', 'MPOW', 'MPPA', 
    'MPXL', 'MRAT', 'MREI', 'MSIN', 'MSKY', 'MTMH', 'MTEL', 'MTLA', 'MTPS', 'MTRA', 'MTSM', 'MYOH', 
    'MYOR', 'MYRX', 'MYTX',
    'NASA', 'NATO', 'NCKL', 'NELY', 'NESR', 'NETV', 'NFCX', 'NICK', 'NICL', 'NIKL', 'NIPS', 'NIRO', 
    'NISP', 'NOBU', 'NPGF', 'NRCA', 'NUSA', 'NZIA', 
    'OASA', 'OBMD', 'OCAP', 'OILS', 'OKAS', 'OMRE', 'OPMS', 
    'PADI', 'PALM', 'PAMG', 'PANI', 'PANR', 'PANS', 'PBID', 'PBRX', 'PBSA', 'PCAR', 'PDES', 'PEGE', 
    'PEHA', 'PGAS', 'PGLI', 'PGUN', 'PICO', 'PJAA', 'PKPK', 'PLIN', 'PLKS', 'PMJS', 'PMMP', 'PNBN', 
    'PNBS', 'PNLF', 'PNSE', 'POLA', 'POLI', 'POLL', 'POLY', 'POOL', 'PORT', 'POWR', 'PPRE', 'PPRO', 
    'PRAS', 'PRIM', 'PSAB', 'PSCS', 'PSDN', 'PSGO', 'PSKT', 'PSSI', 'PTBA', 'PTDU', 'PTIS', 'PTPP', 
    'PTRO', 'PTSN', 'PTSP', 'PUDP', 'PURA', 'PUSA', 'PYFA', 'PZZA',
    'RAJA', 'RALS', 'RANC', 'RBMS', 'RCY', 'RDTX', 'REAL', 'RELI', 'RISE', 'RMBA', 'RMKE', 'ROCK', 
    'RODA', 'RONY', 'ROTI', 'RSGK', 'RUIS', 'RUNS', 
    'SAFE', 'SAME', 'SAMF', 'SAPX', 'SATU', 'SBAT', 'SCCO', 'SCMA', 'SCNP', 'SCPI', 'SDPC', 'SDRA', 
    'SDMU', 'SDO', 'SEMA', 'SFAN', 'SGER', 'SGRO', 'SHID', 'SHIP', 'SIDO', 'SILO', 'SIMA', 'SIMP', 
    'SINI', 'SIPD', 'SKBM', 'SKLT', 'SKRN', 'SKYB', 'SLIS', 'SMAR', 'SMBR', 'SMCB', 'SMDM', 'SMDR', 
    'SMGR', 'SMKL', 'SMMA', 'SMRA', 'SMSM', 'SNLK', 'SOBI', 'SOCI', 'SOFA', 'SOHO', 'SONA', 'SOSS', 
    'SOTS', 'SPMA', 'SPTO', 'SQMI', 'SRAJ', 'SRIL', 'SRSN', 'SRTG', 'SSTM', 'STAR', 'STTP', 'SUDI', 
    'SUGI', 'SULI', 'SUPR', 'SURE', 'SWAT',
    'TALF', 'TAMA', 'TAMU', 'TAPG', 'TARA', 'TAXI', 'TBIG', 'TBLA', 'TBMS', 'TCID', 'TCPI', 'TDPM', 
    'TEBE', 'TECH', 'TELE', 'TFAS', 'TFCO', 'TGKA', 'TGRA', 'TIFA', 'TINS', 'TIRA', 'TIRT', 'TKIM', 
    'TLKM', 'TMAS', 'TMPO', 'TNCA', 'TOBA', 'TOPS', 'TOTL', 'TOWR', 'TPIA', 'TPMA', 'TRIL', 'TRIM', 
    'TRIN', 'TRIS', 'TRJA', 'TRST', 'TRUK', 'TRUS', 'TSPC', 'TUGU', 'TURI', 
    'ULTJ', 'UNIC', 'UNIQ', 'UNIT', 'UNSP', 'UNTR', 'UNVR', 'UANG', 'URAY', 'UVCR', 
    'VICF', 'VICI', 'VINS', 'VIVA', 'VOKS', 'VRNA', 
    'WAPO', 'WEGE', 'WEHA', 'WGSH', 'WICO', 'WIFI', 'WIKA', 'WINS', 'WOMF', 'WOOD', 'WOWS', 'WSKT', 
    'WTON', 
    'YELO', 'YPAS', 'YULE', 
    'ZBRA', 'ZINC', 'ZONE', 'ZYRX'
]

# Hapus duplikat dan TAMBAHKAN .JK
ALL_TICKERS = sorted(list(set([f"{t}.JK" for t in RAW_TICKERS])))

BATCH_SIZE = 20 # Saya naikkan batch jadi 20 biar agak cepat

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Buat tabel
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database siap/terhubung.")

def update_data():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    total_saham = len(ALL_TICKERS)
    print(f"🚀 Memulai update untuk {total_saham} saham Indonesia...")
    
    total_sukses = 0
    
    # Teknik Batch Processing
    for i in range(0, total_saham, BATCH_SIZE):
        batch = ALL_TICKERS[i:i+BATCH_SIZE]
        str_tickers = " ".join(batch)
        print(f"📥 [{i+1}/{total_saham}] Mengunduh: {batch[0]} s/d {batch[-1]} ...")
        
        try:
            # Ambil data 1 tahun terakhir
            df = yf.download(str_tickers, period="1y", interval="1d", group_by='ticker', progress=False)
            
            for ticker in batch:
                try:
                    if len(batch) > 1:
                        data = df[ticker].copy()
                    else:
                        data = df.copy()
                    
                    if data.empty: continue
                    
                    data.reset_index(inplace=True)
                    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
                    
                    records = []
                    for _, row in data.iterrows():
                        records.append((
                            ticker, 
                            row['Date'], 
                            row['Open'], 
                            row['High'], 
                            row['Low'], 
                            row['Close'], 
                            row['Volume']
                        ))
                    
                    conn.executemany('''
                        INSERT OR REPLACE INTO daily_prices 
                        (ticker, date, open, high, low, close, volume) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', records)
                    
                    total_sukses += 1
                    
                except Exception:
                    continue
            
            conn.commit()
            time.sleep(0.5) # Jeda sedikit biar aman
            
        except Exception as e:
            print(f"❌ Batch Error: {e}")

    conn.close()
    print(f"\n🎉 SELESAI! {total_sukses} saham berhasil disimpan ke '{DB_NAME}'.")

if __name__ == "__main__":
    update_data()