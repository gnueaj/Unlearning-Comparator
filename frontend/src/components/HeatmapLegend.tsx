import { useRef, useEffect } from "react";
import * as d3 from "d3";

const width = 8;
const height = 188;
const margin = { top: 10, right: 30, bottom: 10, left: 0 };

export default function HeatmapLegend() {
  const legendRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (legendRef.current) {
      const svg = d3
        .select(legendRef.current)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

      const defs = svg.append("defs");
      const linearGradient = defs
        .append("linearGradient")
        .attr("id", "legend-gradient")
        .attr("x1", "0%")
        .attr("y1", "100%")
        .attr("x2", "0%")
        .attr("y2", "0%");

      const stops = [
        { offset: "0%", color: d3.interpolateViridis(0) },
        { offset: "100%", color: d3.interpolateViridis(1) },
      ];

      linearGradient
        .selectAll("stop")
        .data(stops)
        .enter()
        .append("stop")
        .attr("offset", (d) => d.offset)
        .attr("stop-color", (d) => d.color);

      svg
        .append("rect")
        .attr("x", margin.left)
        .attr("y", margin.top)
        .attr("width", width)
        .attr("height", height)
        .style("fill", "url(#legend-gradient)");

      const legendScale = d3
        .scaleLinear()
        .domain([0, 1])
        .range([height + margin.top, margin.top]);

      const legendAxis = d3
        .axisRight(legendScale)
        .ticks(5)
        .tickFormat(d3.format(".1f"))
        .tickSize(4)
        .tickPadding(6);

      svg
        .append("g")
        .attr("class", "legend-axis")
        .attr("transform", `translate(${width + margin.left}, 0)`)
        .call(legendAxis)
        .select(".domain")
        .remove();

      svg
        .selectAll(".legend-axis text")
        .style("font-size", "8px")
        .style("overflow", "visible");
    }
  }, []);

  return (
    <svg
      className="mb-[32.5px] -ml-1"
      ref={legendRef}
      width={width + margin.left + margin.right}
      height={height + margin.top + margin.bottom}
    ></svg>
  );
}