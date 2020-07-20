var smg_person = "https://collection.sciencemuseumgroup.org.uk/people/";
var smg_object = "https://collection.sciencemuseumgroup.org.uk/objects/";
var wikidata_entity = "http://www.wikidata.org/entity";

function filterNodesById(nodes,id){
    return nodes.filter(function(n) { return n.id === id; });
}

function triplesToGraph(triples){

    svg.html("");
    //Graph
    var graph={nodes:[], links:[]};
    
    //Initial Graph from triples
    triples.forEach(function(triple){
        var subjId = triple.subject;
        var predId = triple.predicate;
        var objId = triple.object;
        
        var subjNode = filterNodesById(graph.nodes, subjId)[0];
        var objNode  = filterNodesById(graph.nodes, objId)[0];
        
        if(subjNode==null){
            subjNode = {id:subjId, label:subjId, weight:1};
            graph.nodes.push(subjNode);
        }
        
        if(objNode==null){
            objNode = {id:objId, label:objId, weight:1};
            graph.nodes.push(objNode);
        }
    
        
        graph.links.push({source:subjNode, target:objNode, predicate:predId, weight:1});
    });
    
    return graph;
}


function update(graph){
    // ==================== Add Marker ====================
    svg.append("svg:defs").selectAll("marker")
        .data(["end"])
      .enter().append("svg:marker")
        .attr("id", String)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 30)
        .attr("refY", -0.5)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
      .append("svg:polyline")
        .attr("points", "0,-5 10,0 0,5")
        ;
        
    // ==================== Add Links ====================
    var links = svg.selectAll(".link")
                        .data(graph.links)
                        .enter()
                        .append("line")
                            .attr("marker-end", "url(#end)")
                            .attr("class", "link")
                            .attr("stroke-width",1)
                    ;//links
    
    // ==================== Add Link Names =====================
    var linkTexts = svg.selectAll(".link-text")
                .data(graph.links)
                .enter()
                .append("text")
                    .attr("class", "link-text")
                    .text( function (d) { return ""; })
                ;

        //linkTexts.append("title")
        //		.text(function(d) { return d.predicate; });
                
    // ==================== Add Link Names =====================
    var nodeTexts = svg.selectAll(".node-text")
                .data(graph.nodes)
                .enter()
                .append("text")
                    .attr("class", function (d) {
                        if (d.label.startsWith(smg_person) | d.label.startsWith(smg_object) | d.label.startsWith(wikidata_entity)) {
                            return "node-text-item"
                        } else{
                            return "node-text"
                        }
                    })
                    .text( function (d) { 
                        if (d.label.startsWith(smg_person) | d.label.startsWith(smg_object) | d.label.startsWith(wikidata_entity)){
                            // var regex = /cp\d+/g;
                            // return regex.exec(d.label)
                            return ""
                        } else {
                            return d.label
                        }
                     })
                    // .text("")
                ;

        //nodeTexts.append("title")
        //		.text(function(d) { return d.label; });
    
    // ==================== Add Node =====================
    var nodes = svg.selectAll(".node")
                        .data(graph.nodes)
                        .enter()
                        .append("circle")
                            .attr("class", function (d) {
                                if (d.label.startsWith(smg_person)){
                                    return "node-smg-person"
                                } 
                                else if (d.label.startsWith(smg_object)) {
                                    return "node-smg-object"
                                }
                                else if (d.label.startsWith(wikidata_entity)) {
                                    return "node-wikidata"
                                }
                                 else {
                                    return "node"
                                }
                            })
                            .attr("r", 4)
                            .call(force.drag)
                    ;//nodes

    // ==================== Force ====================
    force.on("tick", function() {
        nodes
            .attr("cx", function(d){ return d.x; })
            .attr("cy", function(d){ return d.y; })
            ;
        
        links
            .attr("x1", 	function(d)	{ return d.source.x; })
            .attr("y1", 	function(d) { return d.source.y; })
            .attr("x2", 	function(d) { return d.target.x; })
            .attr("y2", 	function(d) { return d.target.y; })
           ;
           
        nodeTexts
            .attr("x", function(d) { return d.x + 12 ; })
            .attr("y", function(d) { return d.y + 3; })
            ;
            

        linkTexts
            .attr("x", function(d) { return 4 + (d.source.x + d.target.x)/2  ; })
            .attr("y", function(d) { return 4 + (d.source.y + d.target.y)/2 ; })
            ;
    });
    
    // ==================== Run ====================
    var k = Math.sqrt(graph.nodes.length / (w * h));

    force
      .nodes(graph.nodes)
      .links(graph.links)
      .charge(-5/k)
      .gravity(20*k)
      .linkDistance(4)
      .start()
      ;
}

