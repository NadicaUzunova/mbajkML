import React, { FC, useContext, useEffect, useState } from "react";
import './bikestand.css';
import { BikeStandContext } from "../context/bikeStandsContext";
import BikeIcon from '../assets/bike-icon.png';
import { mbajkApi } from "../utils/axios";
import { v4 as uuid } from 'uuid';

/** Pridobi API URL iz okolja */
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

type BikeStandProps = {};
type DataRow = (string | number)[];

const BikeStand: FC<BikeStandProps> = () => {
    const [hours, setHours] = useState<number[]>([]);
    const [data, setData] = useState<number | null>(null);
    const [predictions, setPredictions] = useState<number[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const { currentBikeStand } = useContext(BikeStandContext);

    /** Normalizacija imen postaj */
    const normalizeString = (inputStr: string): string => {
        return inputStr.replace(/[ščžŠČŽ]/g, char => ({
            'š': 's', 'č': 'c', 'ž': 'z',
            'Š': 'S', 'Č': 'C', 'Ž': 'Z'
        }[char] || char));
    };

    /** Izračun naslednjih 7 ur za prikaz napovedi */
    const calculateNextSevenHours = () => {
        const currentHour = new Date().getHours();
        const nextHours = Array.from({ length: 7 }, (_, i) => (currentHour + i + 1) % 24);
        setHours(nextHours);
    };

    /** Pridobi podatke v realnem času in napovedi */
    const fetchPredictions = async () => {
        if (!currentBikeStand) return;

        setIsLoading(true);
        try {
            const location = normalizeString(currentBikeStand.name || currentBikeStand.location);
            console.log("Fetching data for:", location);

            // Pridobi trenutne podatke
            const liveDataRes = await mbajkApi.post(`${API_URL}/live-data`, { location });
            const liveData = liveDataRes.data;
            setData(liveData.available_bikes || 0);

            // Pridobi pretekle podatke
            const oldDataRes = await mbajkApi.post(`${API_URL}/data`, { location });
            let oldData: DataRow[] = oldDataRes.data.data;
            console.log("Old data fetched:", oldData);

            // Očisti podatke (pretvori v številke, nadomesti null vrednosti)
            const cleanOldData = oldData.map(row =>
                row.map(value => (typeof value === 'string' && !isNaN(Number(value)) ? Number(value) : value || 0))
            );

            // Preveri, ali imamo dovolj podatkov za napoved
            if (cleanOldData.length !== 12) {
                console.warn("Insufficient historical data, skipping prediction.");
                setPredictions([]);
                return;
            }

            // Pošlji poizvedbo za napoved
            const predictionRes = await mbajkApi.post(`${API_URL}/mbajk/predict`, {
                location,
                data: cleanOldData,
            });

            setPredictions(predictionRes.data.predictions);
        } catch (error) {
            console.error("Error fetching predictions:", error);
            setPredictions([]);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        calculateNextSevenHours();
        fetchPredictions();
    }, [currentBikeStand]);

    return (
        <div className="bike-stand-menu-container">
            <div className="bike-stand-inner-container">
                {currentBikeStand?.location && (
                    <>
                        <div className="bike-stand-menu-title">
                            {currentBikeStand.location}
                        </div>

                        <div className="bike-stand-current-state-title">Trenutno stanje:</div>
                        <div className="bike-stand-current-state">
                            {isLoading ? (
                                <div className="loading-spinner"></div>
                            ) : (
                                <div className="bike-stand-current-state-number">
                                    {data !== null ? data : "N/A"}
                                </div>
                            )}
                            <div className="bike-stand-current-state-image">
                                <img src={BikeIcon} width="20px" height="20px" alt="Bike icon" />
                            </div>
                        </div>

                        <h3>Napovedi</h3>
                        <div className="date-grid-container">
                            {hours.map((hour, index) => (
                                <div key={`hour-${index}`} className="date-grid-item">{hour}:00</div>
                            ))}

                            {predictions.length > 0 ? (
                                predictions.map((prediction) => (
                                    <div key={uuid()} className="date-grid-item">
                                        <div className="date-grid-item-text">{prediction}</div>
                                        <div className="date-grid-item-image">
                                            <img src={BikeIcon} width="20px" height="20px" alt="Bike icon" />
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="no-data-message">❌ Ni razpoložljivih napovedi.</div>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default BikeStand;