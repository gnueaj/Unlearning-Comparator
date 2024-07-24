import React, { useState, useEffect } from "react";
import styles from "./Embeddings.module.css";

import Title from "../components/Title";
import ContentBox from "../components/ContentBox";
import SubTitle from "../components/SubTitle";

type PropsType = {
  svgContents: string[];
};

export default function Settings({ svgContents }: PropsType) {
  const [modifiedSvgs, setModifiedSvgs] = useState<string[]>([]);
  const [selectedId, setSelectedId] = useState<number | undefined>();

  useEffect(() => {
    const modifySvg = (svg: string) => {
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(svg, "image/svg+xml");
      const legend = svgDoc.getElementById("legend_1");

      if (!legend || !legend.parentNode) return;
      legend.parentNode.removeChild(legend);

      return new XMLSerializer().serializeToString(svgDoc);
    };

    const modified = svgContents.map(modifySvg) as string[];
    setModifiedSvgs(modified);
    setSelectedId(svgContents ? 4 : undefined);
  }, [svgContents]);

  const createMarkup = (svg: string) => {
    return { __html: svg };
  };

  const handleThumbnailClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const id = e.currentTarget.id;
    setSelectedId(parseInt(id));
  };

  return (
    <section>
      <Title title="Embeddings" />
      <div className={styles.section}>
        <ContentBox height={495}>
          <div className={styles.wrapper}>
            <SubTitle subtitle="Original Model" />
            {svgContents && (
              <div className={styles["content-wrapper"]}>
                <div className={styles["svg-wrapper"]}>
                  <div
                    id="1"
                    onClick={handleThumbnailClick}
                    className={styles.svg}
                    dangerouslySetInnerHTML={createMarkup(modifiedSvgs[0])}
                  />
                  <div
                    id="2"
                    onClick={handleThumbnailClick}
                    className={styles.svg}
                    dangerouslySetInnerHTML={createMarkup(modifiedSvgs[1])}
                  />
                  <div
                    id="3"
                    onClick={handleThumbnailClick}
                    className={styles.svg}
                    dangerouslySetInnerHTML={createMarkup(modifiedSvgs[2])}
                  />
                  <div
                    id="4"
                    onClick={handleThumbnailClick}
                    className={styles.svg}
                    dangerouslySetInnerHTML={createMarkup(modifiedSvgs[3])}
                  />
                </div>
                {selectedId && (
                  <div
                    className={styles["selected-svg"]}
                    dangerouslySetInnerHTML={createMarkup(
                      svgContents[selectedId - 1]
                    )}
                  />
                )}
              </div>
            )}
          </div>
        </ContentBox>
        <ContentBox height={495}>
          <div className={styles.wrapper}>
            <SubTitle subtitle="Unlearned Model" />
          </div>
        </ContentBox>
      </div>
    </section>
  );
}
