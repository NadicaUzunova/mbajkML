import React, { createContext, useState } from 'react';

const originalBikeStands: Array<BikeStandType> = [
    {
        location: 'DVORANA TABOR',
        latitude: 46.54994670178013,
        longitude: 15.635611927857214
    },
    {
        location: 'EUROPARK - POBRESKA C.',
        latitude: 46.55375398600962,
        longitude: 15.653024168338694,
        name: 'EUROPARK - POBREŠKA C.'
    },
    {
        location: 'GORKEGA UL. - OS FRANCETA PRESERNA',
        latitude: 46.55064737413038,
        longitude: 15.63905099418856,
        name: 'GORKEGA UL. - OŠ FRANCETA PREŠERNA'
    },
    {
        location: 'GOSPOSVETSKA C. - III. GIMNAZIJA',
        latitude: 46.560651390643294,
        longitude: 15.640699416907651
    },
    {
        location: 'GOSPOSVETSKA C. - TURNERJEVA UL.',
        latitude: 46.56268004332136,
        longitude: 15.630055368391249
    },
    {
        location: 'GOSPOSVETSKA C. - VRBANSKA C.',
        latitude: 46.56144311075555,
        longitude: 15.63721873716768
    },
    {
        location: 'JHMB – DVOETAZNI MOST',
        latitude: 46.554685047513956,
        longitude: 15.659588580388618,
        name: 'JHMB – DVOETAŽNI MOST'
    },
    {
        location: 'KOROSKA C. - KOROSKI VETER',
        latitude: 46.563013158296066,
        longitude: 15.627438435964866,
        name: 'KOROŠKA C. - KOROŠKI VETER'
    },
    {
        location: 'LIDL - KOROSKA C.',
        latitude: 46.565603647048704,
        longitude: 15.62228863646738,
        name: 'LIDL - KOROŠKA C.'
    },
    {
        location: 'LIDL - TITOVA C.',
        latitude: 46.551617048905186,
        longitude: 15.652118539503203
    },
    {
        location: 'LJUBLJANSKA UL. - FOCHEVA',
        latitude: 46.54539115536521,
        longitude: 15.643439161470972
    },
    {
        location: 'LJUBLJANSKA UL. - II. GIMNAZIJA',
        latitude: 46.54925222143673,
        longitude: 15.645443392573403
    },
    {
        location: 'MLADINSKA UL. - TRUBARJEVA UL.',
        latitude: 46.56299953180152,
        longitude: 15.644307638363767
    },
    {
        location: 'MLINSKA UL . - AVTOBUSNA POSTAJA',
        latitude: 46.55911401222174,
        longitude: 15.655023261444471
    },
    {
        location: 'NA POLJANAH - HEROJA SERCERJA',
        latitude: 46.55301628615977,
        longitude: 15.623644388609105,
        name: 'NA POLJANAH - HEROJA ŠERCERJA'
    },
    {
        location: 'NICEHASH - C. PROLETARSKIH BRIGAD',
        latitude: 46.54352062314532,
        longitude: 15.631569401428226
    },
    {
        location: 'NKBM - TRG LEONA STUKLJA',
        latitude: 46.55919984694199,
        longitude: 15.648897210668153,
        name: 'NKBM - TRG LEONA ŠTUKLJA'
    },
    {
        location: 'PARTIZANSKA C. - CANKARJEVA UL.',
        latitude: 46.563068064435804,
        longitude: 15.657708481058028
    },
    {
        location: 'PARTIZANSKA C. - TIC',
        latitude: 46.5603857228824,
        longitude: 15.650601111325965
    },
    {
        location: 'PARTIZANSKA C. - ZELEZNISKA POSTAJA',
        latitude: 46.56227152136734,
        longitude: 15.657302894037661,
        name: 'PARTIZANSKA C. - ŽELEZNIŠKA POSTAJA'
    },
    {
        location: 'POSTA - SLOMSKOV TRG',
        latitude: 46.55881666100552,
        longitude: 15.644124556030485,
        name: 'POŠTA - SLOMŠKOV TRG'
    },
    {
        location: 'RAZLAGOVA UL. - OBCINA',
        latitude: 46.562097705019994,
        longitude: 15.649762316302898,
        name: 'RAZLAGOVA UL. - OBČINA'
    },
    {
        location: 'SPAR - TRZNICA TABOR',
        latitude: 46.56020567623396,
        longitude: 15.64879678932114,
        name: 'SPAR - TRŽNICA TABOR'
    },
    {
        location: 'STROSSMAYERJEVA UL. - TRZNICA',
        latitude: 46.56125505581808,
        longitude: 15.642316310668194,
        name: 'STROSSMAYERJEVA UL. - TRŽNICA'
    },
    {
        location: 'TELEMACH - GLAVNI TRG - STARI PERON',
        latitude: 46.557606819614165,
        longitude: 15.645610341351302
    },
    {
        location: 'ULICA MOSE PIJADA - UKC',
        latitude: 46.55374578603418,
        longitude: 15.642577397174145,
        name: 'ULICA MOŠE PIJADA - UKC'
    },
    {
        location: 'UM FGPA - LENT - SODNI STOLP',
        latitude: 46.55694263876314,
        longitude: 15.64128138498444
    },
    {
        location: 'VZAJEMNA, VARUH ZDRAVJA - BETNAVSKA C.',
        latitude: 46.544956570634675,
        longitude: 15.638984010667533
    }
]

export type BikeStandType = {
    location: string,
    latitude: number,
    longitude: number,
    name ?: string
}

export type BikeStandContextType = {
    bikeStands: Array<BikeStandType>,
    currentBikeStand: BikeStandType | null,
    changeCurrentBikeStand: (newBikeStand: BikeStandType) => void
}

export const BikeStandContext = createContext<BikeStandContextType>({
    bikeStands: [],
    currentBikeStand: null,
    changeCurrentBikeStand: (newBikeStand: BikeStandType) => {}
});

const BikeStandsProvider = ({ children }: any) => {
  const bikeStands: Array<BikeStandType> = originalBikeStands
  const [currentBikeStand, setCurrentBikeStand] = useState<BikeStandType | null>(null);

  const changeCurrentBikeStand = (newBikeStand: BikeStandType) => {
      setCurrentBikeStand(newBikeStand)
  }

  return <BikeStandContext.Provider value={{bikeStands, currentBikeStand, changeCurrentBikeStand}}>{children}</BikeStandContext.Provider>
};

export default BikeStandsProvider;